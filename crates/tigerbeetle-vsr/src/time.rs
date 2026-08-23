//! Port of the `Time` interface from `src/time.zig`.
//!
//! Upstream models this as a vtable over a context pointer; here it is a trait object, which
//! keeps the dynamic dispatch without `unsafe`.
//!
//! DEVIATION: only the abstract `Time` interface is ported so far. The OS implementation
//! (`TimeOS`, `posix/clock_gettime`) belongs to the io/platform layer and lands later.
//! TODO(port): `src/time.zig` `TimeOS`, `Instant` wrappers.

use tigerbeetle_core::stdx::Instant;

/// A time source: system time or deterministic simulation time.
///
/// `monotonic` is a timestamp to measure elapsed time, meaningful only on the same system, not
/// across reboots. Always use a monotonic timestamp if the goal is to measure elapsed time.
/// This clock is not affected by discontinuous jumps in the system time, for example if the
/// system administrator manually changes the clock.
///
/// `realtime` is a timestamp to measure real (i.e. wall clock) time, meaningful across systems,
/// and reboots. This clock is affected by discontinuous jumps in the system time.
pub trait Time {
    fn monotonic(&mut self) -> Instant;

    /// This should never be used by the state machine, only for measuring clock offsets.
    fn realtime(&self) -> i64;

    /// Advances simulated time; a no-op for real clocks (upstream: `tick`).
    fn tick(&mut self);
}
