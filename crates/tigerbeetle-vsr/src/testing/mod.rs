//! Port of deterministic time simulation from `src/testing/time.zig`.
//!
//! Used by unit tests and simulators to drive `Clock` with controlled clock offsets.

pub mod time;

pub use time::{OffsetType, TimeSim};
