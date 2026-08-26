//! Port of `src/vsr/message_bus.zig` — manages TCP connections between replicas
//! and between clients and the cluster.
//!
//! DEVIATION: upstream is a comptime-parameterized `MessageBusType(IO)` where `IO`
//! is the platform-specific backend. This port uses the [`crate::io::Io`] trait
//! instead, and the bus is generic over `I: Io`.
//!
//! The real async event-loop integration is deferred.  This module provides the
//! data structures and connection-management logic; the tick/recv/send loops
//! will be wired once a proper async runtime is in place.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};

use tigerbeetle_core::constants;

use crate::io::{self, Completion, Io, ListenOptions, SynchronousIo};
use crate::message::Message;
use crate::message_pool::MessagePool;

// ---------------------------------------------------------------------------
// Process identity
// ---------------------------------------------------------------------------

/// Identifies whether this process is a replica or a client.
///
/// Upstream: `src/vsr/message_bus.zig:24` (`ProcessID`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Process {
    Replica { index: u8 },
    Client { id: u128 },
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// State of a single TCP connection.
///
/// Upstream: `Connection` struct in `src/vsr/message_bus.zig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    Free,
    Connecting,
    Connected,
    Accepting,
    Terminating,
}

/// A single TCP connection slot.
///
/// Upstream: `Connection` in `src/vsr/message_bus.zig:1258`.
pub struct Connection {
    pub state: ConnectionState,
    pub peer: Peer,
    pub stream: Option<TcpStream>,
    pub send_queue: Vec<Message>,
    pub recv_buffer: Vec<u8>,
}

use crate::Peer;

impl Connection {
    /// Create a free (unused) connection slot.
    #[must_use]
    pub fn free() -> Self {
        Self {
            state: ConnectionState::Free,
            peer: Peer::Unknown,
            stream: None,
            send_queue: Vec::new(),
            recv_buffer: Vec::with_capacity(constants::MESSAGE_SIZE_MAX as usize),
        }
    }
}

// ---------------------------------------------------------------------------
// MessageBus
// ---------------------------------------------------------------------------

/// The message bus manages all TCP connections for a single process (replica or client).
///
/// Upstream: `src/vsr/message_bus.zig:29` (`MessageBusType`).
///
/// DEVIATION: upstream stores raw fd handles and manages the event loop via IO;
/// this port holds `TcpStream` objects and defers the tick/recv/send loops to a
/// future async integration.
pub struct MessageBus {
    /// Identity of this process.
    pub process: Process,
    /// Shared message pool for allocating messages.
    pub message_pool: MessagePool,
    /// Fixed-size pool of connection slots.
    connections: Vec<Connection>,
    /// Maps replica index → active connection slot.
    replicas: Vec<Option<usize>>,
    /// Maps client id → active connection slot.
    #[allow(dead_code)]
    clients: HashMap<u128, usize>,
    /// Listening socket (replicas only).
    listener: Option<TcpListener>,
    /// Addresses of other replicas.
    replica_addresses: Vec<SocketAddr>,
}

impl MessageBus {
    /// Create a new message bus for a replica process.
    ///
    /// # Panics
    /// Panics if `replica_count > REPLICAS_MAX` (upstream asserts).
    #[must_use]
    pub fn new_replica(
        replica_index: u8,
        replica_addresses: Vec<SocketAddr>,
        pool: MessagePool,
    ) -> Self {
        assert!(
            (replica_index as usize) < constants::REPLICAS_MAX,
            "replica_index must be < REPLICAS_MAX"
        );

        // Upstream: connections_max = configuration.len - 1 + clients_limit + 1.
        // We don't know clients_limit yet, so use MEMBERS_MAX + 1 as a safe upper bound.
        let connections_max = constants::MEMBERS_MAX + 1;
        let mut connections = Vec::with_capacity(connections_max);
        for _ in 0..connections_max {
            connections.push(Connection::free());
        }

        let mut replicas = Vec::with_capacity(constants::REPLICAS_MAX);
        replicas.resize(constants::REPLICAS_MAX, None);

        Self {
            process: Process::Replica { index: replica_index },
            message_pool: pool,
            connections,
            replicas,
            clients: HashMap::new(),
            listener: None,
            replica_addresses,
        }
    }

    /// Create a new message bus for a client process.
    #[must_use]
    pub fn new_client(client_id: u128, pool: MessagePool) -> Self {
        // Upstream: client connections_max = configuration.len (one per replica).
        let connections_max = constants::REPLICAS_MAX;
        let mut connections = Vec::with_capacity(connections_max);
        for _ in 0..connections_max {
            connections.push(Connection::free());
        }

        Self {
            process: Process::Client { id: client_id },
            message_pool: pool,
            connections,
            replicas: Vec::new(),
            clients: HashMap::new(),
            listener: None,
            replica_addresses: Vec::new(),
        }
    }

    /// Begin listening for incoming connections (replicas only).
    ///
    /// # Errors
    /// Returns an error if the socket cannot be bound or if this is a client process.
    pub fn listen(&mut self, address: &str, port: u16) -> std::io::Result<()> {
        if matches!(self.process, Process::Client { .. }) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "clients do not listen",
            ));
        }
        let listener = io::listen(address, port, ListenOptions::default())?;
        self.listener = Some(listener);
        Ok(())
    }

    /// The replica index (only valid for replica processes).
    #[must_use]
    pub fn replica_index(&self) -> Option<u8> {
        match self.process {
            Process::Replica { index } => Some(index),
            Process::Client { .. } => None,
        }
    }

    /// Whether this process is the primary for the current view.
    #[must_use]
    pub fn is_replica(&self) -> bool {
        matches!(self.process, Process::Replica { .. })
    }

    /// Find a free connection slot, or reclaim one if all are occupied.
    fn allocate_connection(&mut self) -> Option<usize> {
        // First: find a free slot.
        if let Some(index) = self.connections.iter().position(|c| c.state == ConnectionState::Free)
        {
            return Some(index);
        }
        // Upstream: reclaim_connection terminates the oldest unknown peer, then any client,
        // then any unknown peer.  For now, return None (full pool).
        None
    }

    /// Send a message to a specific replica.
    ///
    /// # Panics
    /// Panics if `target >= REPLICAS_MAX`.
    ///
    /// # Errors
    /// Returns an error if no connection can be established.
    pub fn send_message_to_replica<I: Io>(
        &mut self,
        io: &I,
        target: u8,
        message: Message,
    ) -> std::io::Result<()> {
        assert!((target as usize) < constants::REPLICAS_MAX, "target must be < REPLICAS_MAX");

        let slot = if let Some(slot) = self.replicas.get(target as usize).and_then(|&s| s) {
            slot
        } else {
            // Open a new connection to this replica.
            let addr = self.replica_addresses.get(target as usize).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "replica address not configured")
            })?;
            let mut comp = Completion::success(0);
            let stream = io.connect(addr, None, &mut comp).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    "could not connect to replica",
                )
            })?;
            let slot = self
                .allocate_connection()
                .ok_or_else(|| std::io::Error::other("connection pool full"))?;
            let conn = &mut self.connections[slot];
            conn.state = ConnectionState::Connected;
            conn.peer = Peer::Replica { replica: target };
            conn.stream = Some(stream);
            if let Some(s) = self.replicas.get_mut(target as usize) {
                *s = Some(slot);
            }
            slot
        };

        let conn = &mut self.connections[slot];
        conn.send_queue.push(message);
        Ok(())
    }

    /// Receive pending messages from all connected peers.
    ///
    /// Returns a list of (peer, message) pairs.
    ///
    /// DEVIATION: upstream uses async IO with completion callbacks; this stub
    /// performs blocking reads for integration testing.
    #[allow(clippy::missing_panics_doc)]
    pub fn receive_messages(&mut self) -> Vec<(Peer, Message)> {
        for slot in 0..self.connections.len() {
            let conn = &mut self.connections[slot];
            if conn.state != ConnectionState::Connected {
                continue;
            }
            let Some(stream) = &conn.stream else {
                continue;
            };

            let mut buf = vec![0u8; constants::MESSAGE_SIZE_MAX as usize];
            let mut comp = Completion::success(0);
            let n = {
                let io = SynchronousIo;
                io.recv(stream, &mut buf, &mut comp);
                comp.result
            };
            if n > 0 {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::unwrap_used
                )]
                let len = usize::try_from(n).unwrap();
                conn.recv_buffer.extend_from_slice(&buf[..len]);
                // TODO(port): parse header, verify checksum, hand off to Replica.
            }
        }
        Vec::new()
    }

    /// Drain the send queue for a connection, writing messages to the socket.
    pub fn flush_send_queue(&mut self) {
        for slot in 0..self.connections.len() {
            let conn = &mut self.connections[slot];
            if conn.state != ConnectionState::Connected {
                continue;
            }
            let Some(stream) = &conn.stream else {
                continue;
            };
            while let Some(message) = conn.send_queue.first() {
                let io = SynchronousIo;
                let mut comp = Completion::success(0);
                io.send(stream, message.buffer(), &mut comp);
                if comp.is_ok() {
                    conn.send_queue.remove(0);
                } else {
                    break;
                }
            }
        }
    }

    /// Get the peer identity for a connection slot.
    #[must_use]
    pub fn peer(&self, slot: usize) -> Peer {
        self.connections.get(slot).map_or(Peer::Unknown, |c| c.peer)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::message_pool::{self, MessageBus as MessageBusType, Options};

    fn test_pool() -> MessagePool {
        MessagePool::new(&Options::Replica(message_pool::Replica {
            members_count: 3,
            pipeline_requests_limit: 1,
            message_bus: MessageBusType::Testing,
        }))
    }

    #[test]
    fn message_bus_new_replica() {
        let pool = test_pool();
        let bus = MessageBus::new_replica(0, vec![], pool);
        assert!(bus.is_replica());
        assert_eq!(bus.replica_index(), Some(0));
    }

    #[test]
    fn message_bus_new_client() {
        let pool = test_pool();
        let bus = MessageBus::new_client(42, pool);
        assert!(!bus.is_replica());
        assert_eq!(bus.replica_index(), None);
    }

    #[test]
    fn allocate_connection_returns_first_free() {
        let pool = test_pool();
        let mut bus = MessageBus::new_replica(0, vec![], pool);
        let slot = bus.allocate_connection();
        assert!(slot.is_some());
        assert_eq!(slot.unwrap(), 0);
    }

    #[test]
    fn allocate_connection_exhausts_pool() {
        let pool = MessagePool::init_capacity(2);
        let mut bus = MessageBus::new_replica(0, vec![], pool);
        // All slots start Free; fill them all.
        let total = bus.connections.len();
        for i in 0..total {
            let s = bus.allocate_connection().unwrap();
            assert_eq!(s, i);
            bus.connections[s].state = ConnectionState::Connected;
        }
        // Next should fail.
        assert!(bus.allocate_connection().is_none());
    }
}
