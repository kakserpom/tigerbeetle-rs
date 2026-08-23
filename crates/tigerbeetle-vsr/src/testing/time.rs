//! Port of `src/testing/time.zig` (`TimeSim`).

use tigerbeetle_core::stdx::Instant;

use crate::time::Time;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetType {
    Linear,
    Periodic,
    Step,
    // TODO(port): src/testing/time.zig OffsetType.non_ideal — needs PRNG.floatNorm(); upstream
    // unit tests never exercise it ("use ideal clocks for the unit tests").
}

/// A deterministic simulated time source.
///
/// * `Linear` offset is described as `A * x + B`: A is the drift per tick and B the initial
///   offset.
/// * `Periodic` is described as `A * sin(x * pi / B)`: A controls the amplitude and B the period
///   in terms of ticks.
/// * `Step` represents a discontinuous jump in the wall-clock time. B is the tick at which the
///   jump occurs. A is the amplitude of the step.
#[derive(Clone, Debug)]
pub struct TimeSim {
    /// The duration of a single tick in nanoseconds.
    pub resolution: u64,

    pub offset_type: OffsetType,

    /// Co-efficients to scale the offset according to the `offset_type`.
    pub offset_coefficient_a: i64,
    pub offset_coefficient_b: i64,

    /// The number of ticks elapsed since initialization.
    pub ticks: u64,

    /// The instant in time chosen as the origin of this time source.
    pub epoch: i64,
}

impl TimeSim {
    #[must_use]
    pub fn new(resolution: u64, offset_type: OffsetType, a: i64, b: i64) -> Self {
        Self {
            resolution,
            offset_type,
            offset_coefficient_a: a,
            offset_coefficient_b: b,
            ticks: 0,
            epoch: 0,
        }
    }

    /// The wall-clock error injected by this simulation at the given tick.
    ///
    /// Note: like upstream, this uses floating point for the periodic variant. This is test-only
    /// simulation scaffolding; the state machine itself remains float-free.
    #[must_use]
    pub fn offset(&self, ticks: u64) -> i64 {
        match self.offset_type {
            OffsetType::Linear => {
                let drift_per_tick = self.offset_coefficient_a;
                // Upstream casts ticks (u64) through i64; bounded by simulation lengths.
                #[allow(clippy::cast_possible_wrap)]
                let ticks = ticks as i64;
                ticks.wrapping_mul(drift_per_tick).wrapping_add(self.offset_coefficient_b)
            }
            OffsetType::Periodic => {
                #[allow(clippy::cast_precision_loss)]
                let ticks = ticks as f64;
                #[allow(clippy::cast_precision_loss)]
                let period = self.offset_coefficient_b as f64;
                #[allow(clippy::cast_precision_loss)]
                let amplitude = self.offset_coefficient_a as f64;
                let unscaled = (ticks * 2.0 * std::f64::consts::PI / period).sin();
                let scaled = amplitude * unscaled;
                // floor then cast: values are bounded by the coefficient magnitude.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let floored = scaled.floor() as i64;
                floored
            }
            OffsetType::Step => {
                if ticks > u64::try_from(self.offset_coefficient_b).unwrap_or(u64::MAX) {
                    self.offset_coefficient_a
                } else {
                    0
                }
            }
        }
    }
}

impl Time for TimeSim {
    fn monotonic(&mut self) -> Instant {
        Instant { ns: self.ticks * self.resolution }
    }

    fn realtime(&self) -> i64 {
        let monotonic_ns = self.ticks * self.resolution;
        // Bounded by simulation parameters (same cast as upstream @intCast).
        #[allow(clippy::cast_possible_wrap)]
        let monotonic_ns = monotonic_ns as i64;
        self.epoch.wrapping_add(monotonic_ns).wrapping_sub(self.offset(self.ticks))
    }

    fn tick(&mut self) {
        self.ticks += 1;
    }
}

/// An aliasable handle to a [`TimeSim`], mirroring upstream's pattern where both the test
/// container and the `Clock` hold vtable references to the same simulation state.
///
/// DEVIATION: upstream aliases via raw context pointers; here aliasing goes through
/// `Rc<RefCell<..>>` (no `unsafe`).
#[derive(Clone)]
pub struct SharedTimeSim(std::rc::Rc<std::cell::RefCell<TimeSim>>);

impl SharedTimeSim {
    #[must_use]
    pub fn new(sim: TimeSim) -> Self {
        Self(std::rc::Rc::new(std::cell::RefCell::new(sim)))
    }

    /// Read-only access for assertions.
    pub fn with<R>(&self, f: impl FnOnce(&TimeSim) -> R) -> R {
        f(&self.0.borrow())
    }

    /// Mutating access for driving the simulation directly (upstream tests call
    /// `clock.time.tick()` from outside the clock).
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut TimeSim) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
}

impl Time for SharedTimeSim {
    fn monotonic(&mut self) -> Instant {
        let sim = self.0.borrow();
        Instant { ns: sim.ticks * sim.resolution }
    }

    fn realtime(&self) -> i64 {
        let sim = self.0.borrow();
        sim.realtime()
    }

    fn tick(&mut self) {
        self.0.borrow_mut().ticks += 1;
    }
}
