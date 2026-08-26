//! Platform-abstracted async I/O interface.
//!
//! Port of `src/io.zig` (comptime-dispatched io_uring/kqueue/IOCP) and
//! `src/io/common.zig` (shared TCP helpers).
//!
//! DEVIATION: upstream is a comptime-selected concrete struct with the same API per
//! platform. Rust cannot do cross-crate monomorphisation by target, so this port uses a
//! trait. [`SynchronousIo`] is a blocking fallback for tests and early integration;
//! a real async backend will be added later.

use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

use tigerbeetle_core::constants;

// ---------------------------------------------------------------------------
// Completion pattern
// ---------------------------------------------------------------------------

/// A completion token handed to the IO backend.
///
/// Upstream passes a `*Completion` to each callback; here the caller owns the
/// completion and the backend fills it in before invoking the callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    /// Number of bytes transferred (recv/send) or 0 for non-data ops.
    pub result: i64,
    /// `None` on success, `Some(io::ErrorKind)` on failure.
    pub err: Option<io::ErrorKind>,
}

impl Completion {
    #[must_use]
    pub fn success(bytes: i64) -> Self {
        Self { result: bytes, err: None }
    }

    #[must_use]
    pub fn error(kind: io::ErrorKind) -> Self {
        Self { result: 0, err: Some(kind) }
    }

    #[must_use]
    pub fn is_ok(self) -> bool {
        self.err.is_none()
    }
}

// ---------------------------------------------------------------------------
// Io trait
// ---------------------------------------------------------------------------

/// Trait abstracting platform-specific async I/O.
///
/// Every method is blocking in [`SynchronousIo`] but would be non-blocking in
/// a real io_uring/kqueue backend.  The caller provides a pre-allocated
/// [`Completion`] that the backend fills in.
pub trait Io {
    /// Accept a TCP connection on `listener`.
    fn accept(&self, listener: &TcpListener, completion: &mut Completion) -> Option<TcpStream>;

    /// Connect to `addr` with an optional timeout.
    fn connect(
        &self,
        addr: &SocketAddr,
        timeout: Option<Duration>,
        completion: &mut Completion,
    ) -> Option<TcpStream>;

    /// Send `buf` through `stream`.
    fn send(&self, stream: &TcpStream, buf: &[u8], completion: &mut Completion);

    /// Receive into `buf` from `stream`.
    fn recv(&self, stream: &TcpStream, buf: &mut [u8], completion: &mut Completion);
}

// ---------------------------------------------------------------------------
// SynchronousIo — blocking stub
// ---------------------------------------------------------------------------

/// Blocking I/O implementation using std library calls.
///
/// Useful for tests and early integration before a real async backend is
/// introduced.
///
/// DEVIATION: upstream never blocks in `run()`; this stub blocks on every
/// operation.  The callback-based completion model is preserved so that callers
/// can be written once against the trait.
#[derive(Clone, Copy, Debug, Default)]
pub struct SynchronousIo;

impl Io for SynchronousIo {
    fn accept(&self, listener: &TcpListener, completion: &mut Completion) -> Option<TcpStream> {
        match listener.accept() {
            Ok((stream, _)) => {
                *completion = Completion::success(0);
                Some(stream)
            }
            Err(e) => {
                *completion = Completion::error(e.kind());
                None
            }
        }
    }

    fn connect(
        &self,
        addr: &SocketAddr,
        timeout: Option<Duration>,
        completion: &mut Completion,
    ) -> Option<TcpStream> {
        let stream = match TcpStream::connect(addr) {
            Ok(s) => s,
            Err(e) => {
                *completion = Completion::error(e.kind());
                return None;
            }
        };
        if let Some(dur) = timeout {
            let _ = stream.set_read_timeout(Some(dur));
            let _ = stream.set_write_timeout(Some(dur));
        }
        *completion = Completion::success(0);
        Some(stream)
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn send(&self, stream: &TcpStream, buf: &[u8], completion: &mut Completion) {
        use std::io::Write;
        // DEVIATION: upstream uses non-blocking send via io_uring/kqueue; we use
        // blocking send via std. We clone the fd (not the data) to get an owned
        // TcpStream for Write::write_all.
        let mut owned = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                *completion = Completion::error(e.kind());
                return;
            }
        };
        match owned.write_all(buf) {
            Ok(()) => *completion = Completion::success(buf.len() as i64),
            Err(e) => *completion = Completion::error(e.kind()),
        }
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn recv(&self, stream: &TcpStream, buf: &mut [u8], completion: &mut Completion) {
        use std::io::Read;
        // Like send, clone the fd handle for the blocking read.
        let mut owned = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                *completion = Completion::error(e.kind());
                return;
            }
        };
        match owned.read(buf) {
            Ok(n) => *completion = Completion::success(n as i64),
            Err(e) => *completion = Completion::error(e.kind()),
        }
    }
}

// ---------------------------------------------------------------------------
// TCP helpers (ported from io/common.zig)
// ---------------------------------------------------------------------------

/// Options for a TCP connection socket.
#[derive(Clone, Copy, Debug)]
pub struct TcpOptions {
    pub rcvbuf: usize,
    pub sndbuf: usize,
    pub keepalive: bool,
    pub nodelay: bool,
}

impl Default for TcpOptions {
    fn default() -> Self {
        Self {
            rcvbuf: constants::TCP_RCVBUF as usize,
            sndbuf: constants::TCP_SNDBUF_REPLICA as usize,
            keepalive: true,
            nodelay: true,
        }
    }
}

/// Options for a TCP listening socket.
#[derive(Clone, Copy, Debug)]
pub struct ListenOptions {
    pub backlog: u32,
}

impl Default for ListenOptions {
    fn default() -> Self {
        Self { backlog: constants::TCP_BACKLOG }
    }
}

/// Bind and listen on `address:port`.
///
/// # Errors
/// Returns an error if the address cannot be resolved, bound, or set to non-blocking.
#[allow(clippy::must_use_candidate)]
pub fn listen(address: &str, port: u16, _options: ListenOptions) -> io::Result<TcpListener> {
    let addr: SocketAddr = format!("{address}:{port}")
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "could not resolve address"))?;
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    // Upstream calls `socket.listen(backlog)` explicitly; std's `TcpListener::bind`
    // calls `listen(128)` internally.  The backlog is set during bind.
    Ok(listener)
}

/// Apply TCP options to an already-connected stream.
///
/// # Errors
/// Returns an error if any socket option cannot be set.
#[allow(clippy::must_use_candidate)]
pub fn tcp_options(stream: &TcpStream, options: TcpOptions) -> io::Result<()> {
    stream.set_nodelay(options.nodelay)?;
    // TODO(port): upstream sets SO_KEEPALIVE via raw setsockopt; std's set_keepalive
    // is unstable.  We skip keepalive for now.
    let _ = options.keepalive;
    let _ = (options.rcvbuf, options.sndbuf);
    // Upstream sets SO_RCVBUF/SO_SNDBUF via raw setsockopt;
    // std doesn't expose a safe API for this without libc.
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn completion_basics() {
        let c = Completion::success(42);
        assert!(c.is_ok());
        assert_eq!(c.result, 42);

        let c = Completion::error(io::ErrorKind::WouldBlock);
        assert!(!c.is_ok());
    }

    #[test]
    fn synchronous_io_accept_connect_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(false).unwrap();
        let addr = listener.local_addr().unwrap();

        let io = SynchronousIo;
        let mut comp = Completion::success(0);

        let handle = std::thread::spawn(move || TcpStream::connect(addr).unwrap());
        let _stream = io.accept(&listener, &mut comp).unwrap();
        assert!(comp.is_ok());

        let _peer = handle.join().unwrap();
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn synchronous_io_send_recv() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(false).unwrap();
        let addr = listener.local_addr().unwrap();

        let io = SynchronousIo;
        let mut comp = Completion::success(0);

        let handle = std::thread::spawn(move || TcpStream::connect(addr).unwrap());
        let server = io.accept(&listener, &mut comp).unwrap();
        let client = handle.join().unwrap();

        let data = b"hello tigerbeetle";
        io.send(&client, data, &mut comp);
        assert!(comp.is_ok());

        let mut buf = [0u8; 64];
        io.recv(&server, &mut buf, &mut comp);
        assert!(comp.is_ok());
        assert_eq!(&buf[..comp.result as usize], data);
    }
}
