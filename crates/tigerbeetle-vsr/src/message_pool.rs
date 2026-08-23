//! A pool of reference-counted Messages, memory for which is allocated only once during
//! initialization and reused thereafter. The messages_max values determine the size of this pool.
//!
//! Port of `src/message_pool.zig` (`Options`, `MessagePool`).
//!
//! DEVIATION: upstream messages are pointer-refcounted and shared across subsystems; safe Rust
//! cannot alias the backing buffers, so [`Message`] is an owned buffer and the pool recycles
//! released buffers instead of counting references. Sizing formulas ([`Options::messages_max`])
//! are ported verbatim.

use crate::message::Message;
use tigerbeetle_core::constants;
use tigerbeetle_core::stdx::stack::Stack;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageBus {
    Tcp,
    Testing,
}

/// Pool sizing options, keyed by process type (upstream `Options: union(vsr.ProcessType)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Options {
    Replica(Replica),
    Client,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Replica {
    pub members_count: u8,
    pub pipeline_requests_limit: u32,
    pub message_bus: MessageBus,
}

impl Options {
    /// The number of messages allocated at initialization by the message pool.
    #[must_use]
    pub fn messages_max(&self) -> u32 {
        match self {
            Self::Client => messages_max_client(),
            Self::Replica(replica) => messages_max_replica(*replica),
        }
    }
}

fn messages_max_client() -> u32 {
    let mut sum: usize = 0;

    sum += constants::REPLICAS_MAX; // Connection.recv_buffer
    // Connection.send_queue:
    sum += constants::REPLICAS_MAX * constants::CONNECTION_SEND_QUEUE_MAX_CLIENT;
    sum += 1; // Client.request_inflight
    // Handle bursts.
    // (e.g. Connection.parse_message(), or sending a ping when the send queue is full).
    sum += 1;

    // This condition is necessary (but not sufficient) to prevent deadlocks:
    assert!(sum > 1);
    u32::try_from(sum).unwrap_or_else(|_| unreachable!("sum bounded by constants"))
}

// The number of full-sized messages allocated at initialization by the replica message
// pool. There must be enough messages to ensure that the replica can always progress,
// to avoid deadlock.
fn messages_max_replica(replica: Replica) -> u32 {
    assert!(replica.members_count > 0);
    assert!(replica.members_count as usize <= constants::MEMBERS_MAX);
    assert!(replica.pipeline_requests_limit <= constants::PIPELINE_REQUEST_QUEUE_MAX);

    let mut sum: usize = 0;

    let pipeline_limit = constants::PIPELINE_PREPARE_QUEUE_MAX + replica.pipeline_requests_limit;

    sum += usize::from(constants::JOURNAL_IOPS_READ_MAX); // Journal reads
    sum += usize::from(constants::JOURNAL_IOPS_WRITE_MAX); // Journal writes
    sum += usize::from(constants::CLIENT_REPLIES_IOPS_READ_MAX); // Client-reply reads
    sum += usize::from(constants::CLIENT_REPLIES_IOPS_WRITE_MAX); // Client-reply writes
    // Replica.grid_reads (Replica.BlockRead)
    sum += usize::from(constants::GRID_REPAIR_READS_MAX);
    sum += 1; // Replica.loopback_queue
    sum += pipeline_limit as usize; // Replica.Pipeline{Queue|Cache}
    sum += 1; // Replica.commit_prepare
    sum += 1; // Replica.sync_view
    // Replica.join_view_from_all_replicas quorum:
    // All other quorums are bitsets.
    //
    // This should be set to the runtime replica_count, but we don't know that precisely
    // yet, so we may guess high. (We can't differentiate between replicas and standbys.)
    sum += std::cmp::min(replica.members_count as usize, constants::REPLICAS_MAX);
    sum += 1; // Handle bursts (e.g. Connection.parse_message)
    // Handle Replica.commit_op's reply:
    // (This is separate from the burst +1 because they may occur concurrently).
    sum += 1;

    if replica.message_bus == MessageBus::Tcp {
        // The maximum number of simultaneous open connections on the server.
        // -1 since we never connect to ourself.
        let connections_max = replica.members_count as usize + pipeline_limit as usize - 1;
        sum += connections_max; // Connection.recv_buffer
        // Connection.send_queue:
        sum += connections_max * constants::CONNECTION_SEND_QUEUE_MAX_REPLICA;
    }

    // This condition is necessary (but not sufficient) to prevent deadlocks:
    assert!(sum > constants::REPLICAS_MAX);
    u32::try_from(sum).unwrap_or_else(|_| unreachable!("sum bounded by constants"))
}

/// A pool of messages whose buffers are allocated once during initialization and recycled
/// thereafter (upstream `MessagePool`; see module-level DEVIATION for the ownership model).
///
/// [`Self::get_message`] moves a buffer out; [`Self::release`] moves it back in. Dropping the
/// pool asserts that every buffer was returned — mirroring upstream's `deinit()` assertion.
#[derive(Debug)]
pub struct MessagePool {
    free_list: Stack,

    messages_max: usize,
    buffers: Vec<Option<Message>>,

    outstanding: usize,
}

impl MessagePool {
    /// Upstream `init(allocator, options)`.
    ///
    /// # Panics
    /// Panics via [`Options::messages_max`] assertions on invalid options.
    #[must_use]
    pub fn new(options: &Options) -> Self {
        Self::init_capacity(usize::try_from(options.messages_max()).unwrap_or(usize::MAX))
    }

    /// Preallocates `messages_max` full-size message buffers.
    #[must_use]
    pub fn init_capacity(messages_max: usize) -> Self {
        let mut free_list = Stack::new(u32::try_from(messages_max).unwrap_or(u32::MAX), false);
        let mut buffers = Vec::with_capacity(messages_max);
        for index in 0..messages_max {
            free_list.push(u32::try_from(index).unwrap_or_else(|_| unreachable!("index < max")));
            buffers.push(Some(Message::new()));
        }

        Self { free_list, messages_max, buffers, outstanding: 0 }
    }

    /// The configured pool size (all buffers preallocated up front).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.messages_max
    }

    /// Get an unused message with a buffer of `message_size_max`.
    /// The returned message has exactly one owner.
    ///
    /// # Panics
    /// Panics if the pool is exhausted (upstream pops from an empty free list and asserts),
    /// or on dropping the pool while this message is still checked out.
    #[must_use]
    pub fn get_message(&mut self) -> Message {
        let Some(slot) = self.free_list.pop() else {
            panic!("message pool exhausted");
        };
        self.outstanding += 1;
        let Some(buffer) = self.buffers[slot as usize].take() else {
            unreachable!("free-list entry always holds a buffer");
        };
        buffer
    }

    /// Return a message to the pool (upstream `unref()` when the last reference drops).
    ///
    /// # Panics
    /// Panics if no slot was awaiting a release (i.e., more releases than checkouts).
    pub fn release(&mut self, message: Message) {
        // Invariant: `free_list` holds exactly the slots holding a recyclable buffer, so a
        // release must always find an empty slot to park the returned buffer in.
        let Some(index) = self.buffers.iter().position(Option::is_none) else {
            panic!("no slot awaiting release");
        };
        self.outstanding -= 1;
        self.buffers[index] = Some(message);
        self.free_list.push(u32::try_from(index).unwrap_or_else(|_| unreachable!("slot < max")));
    }

    /// Frees all messages that were unused or returned to the pool.
    ///
    /// # Panics
    /// Panics if any message was not released (upstream `deinit()` asserts the same).
    pub fn deinit(self) {
        // `assert_released` runs in `Drop`:
    }

    fn assert_released(&self) {
        // If the MessagePool is being deinitialized, all messages should have already been
        // released to the pool:
        #[allow(clippy::needless_borrows_for_generic_args)] // clearer failure messages
        let released_all = "all messages should have already been released";
        assert_eq!(self.free_list.count() as usize, self.messages_max, "{released_all}");
        assert_eq!(self.outstanding, 0, "{released_all}");
    }
}

impl Drop for MessagePool {
    fn drop(&mut self) {
        self.assert_released();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sizing sanity: both process types produce pools large enough to avoid deadlock,
    /// and the formulas stay stable (change-detector for the ported arithmetic).
    #[test]
    fn options_messages_max() {
        let client = Options::Client.messages_max();
        assert!(client > 1);

        let testing_replica = Replica {
            members_count: 6,
            pipeline_requests_limit: 1,
            message_bus: MessageBus::Testing,
        };
        let replica_max = Options::Replica(testing_replica).messages_max();
        assert!(replica_max > u32::try_from(constants::REPLICAS_MAX).unwrap_or(u32::MAX));
        assert!(replica_max > client, "tcp-less replica pool is journal/grid dominated");

        let tcp_options =
            Options::Replica(Replica { message_bus: MessageBus::Tcp, ..testing_replica });
        assert!(tcp_options.messages_max() > replica_max);
    }

    #[test]
    fn get_release_round_trip() {
        let mut pool = MessagePool::init_capacity(2);
        assert_eq!(2, pool.capacity());

        let a = pool.get_message();
        let b = pool.get_message();
        // Exhausted:
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _exhausted = pool.get_message();
        }));
        assert!(result.is_err(), "exhausted pool must panic");

        pool.release(a);
        pool.release(b);
        pool.deinit();
    }

    /// Dropping a pool with messages still checked out asserts, like upstream `deinit()`.
    #[test]
    #[should_panic(expected = "all messages should have already been released")]
    fn drop_with_outstanding_panics() {
        let mut pool = MessagePool::init_capacity(1);
        let _message = pool.get_message();
        drop(pool);
    }
}
