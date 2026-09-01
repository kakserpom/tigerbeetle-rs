//! Cluster-wide synchronized clock, aggregating timing information from all replicas.
//!
//! Time plays a central role in TigerBeetle data model. Because it is so important, TigerBeetle
//! defines its own time. In other words, we don't use time to drive consensus, we use consensus to
//! drive time!
//!
//! Time is important for the domain of accounting (e.g., pending transfers can expire with time),
//! but it can't be supplied by the client, as its clock can be unreliable. For this reason,
//! TigerBeetle needs to expose a "time service" to the state machine logic.
//!
//! Additionally, TigerBeetle needs to assign some kind of a sequence number to every event in the
//! system, to make it easy to say whether A happened before B or vice versa.
//!
//! Finally, to maintain indices, the LSM tree could benefit from a compact synthetic primary key.
//!
//! Time solves _all_ of these problems at once: each object in TigerBeetle gets tagged with a u64
//! nanosecond-precision creation timestamp. These timestamps are unique across all objects (an
//! Account and a Transfer can never have the same timestamp), consistent with linearization order
//! of the events (earlier events get smaller timestamps), and closely match the real wall-clock
//! time. Timestamps are used as internal synthetic primary keys instead of user-supplied random
//! u128 ids because they are smaller and also expose temporal locality.
//!
//! Implementation:
//!
//! The ultimate source of timestamps is each replica's operating system. This time is backed by a
//! replica-local drifty hardware clock which is periodically synchronized through NTP with high
//! quality clocks elsewhere. Using system time directly as a source of TigerBeetle timestamps
//! doesn't work:
//!
//! First, system time differs across replicas. To solve this problem, only the primary assigns
//! timestamps. Specifically, when the primary converts a request to a prepare, it assigns its
//! current time to the prepare. The state machine then assigns `prepare_timestamp + object_index`
//! as the creation timestamp for each object in a batch.
//!
//! Second, system time is not monotonic: due to NTP it can easily go backwards. To solve this
//! problem, the primary just takes the max between the current time and the previous timestamp
//! used. Notably, this ends up preserving monotonicity across restarts --- it is when replaying
//! past prepares from the WAL that a replica learns about the latest timestamp before restart.
//!
//! Third, replica's system time lacks high availability: if a primary is isolated from NTP servers
//! its local clock can drift significantly. Another problematic scenario is an operator error
//! which incorrectly adjusts primary's local clock to be far in the future, which, due to
//! monotonicity requirement, could render the cluster completely unusable.
//!
//! To solve the last problem, the primary aggregates clock information from the entire cluster and
//! calculates a timestamp value which is consistent with clocks on at least half of the replicas.
//!
//! Sketch of the algorithm:
//!
//! Assume you have six different clocks. Each clock shows a different time. Most are close, but
//! there could be outliers. How do you estimate the "true" time?
//!
//! The key insight is to think in intervals, rather than points. If a clock shows time t and
//! claims error margin Δ, it means the true time is in the [t-Δ;t+Δ] interval. If you have two
//! clocks, you can intersect their intervals to narrow down the true time interval. If the
//! intervals are disjoint, that means that at least one of the clocks is malfunctioning. This gives
//! an algorithm for identifying cluster time --- collect clock measurements from all replicas
//! together with the respective error margins and find an interval which is consistent with at
//! least half of the clocks.
//!
//! Port of `src/vsr/clock.zig`.
//!
//! DEVIATION: upstream logs via `stdx.log` and feeds gauges to `Tracer`; logging/tracing are not
//! ported yet (no-external-deps policy). All log-driven control flow is preserved; only the
//! operator-visible messages are omitted. TODO(port): src/stdx/log.zig, src/trace.zig.
//!
//! DEVIATION: the PacketSimulator-based integration tests at the bottom of upstream clock.zig
//! depend on `src/testing/packet_simulator.zig`, which is not ported yet.
//! TODO(port): src/vsr/clock.zig packet simulator tests.

#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream

use tigerbeetle_core::constants;
use tigerbeetle_core::stdx::Instant;

use crate::marzullo::{Bound, Interval, Marzullo, Tuple};
use crate::time::Time;

const CLOCK_OFFSET_TOLERANCE_MAX: u64 = constants::CLOCK_OFFSET_TOLERANCE_MAX.to_ns();
const EPOCH_MAX: u64 = constants::CLOCK_EPOCH_MAX.to_ns();
const WINDOW_MIN: u64 = constants::CLOCK_SYNCHRONIZATION_WINDOW_MIN.to_ns();
const WINDOW_MAX: u64 = constants::CLOCK_SYNCHRONIZATION_WINDOW_MAX.to_ns();

// Upstream comptime assert: warn threshold must stay a small fraction of the max tolerance.
const _: () = assert!(50 * tigerbeetle_core::stdx::NS_PER_MS < CLOCK_OFFSET_TOLERANCE_MAX / 100);

#[derive(Clone, Copy, Debug)]
struct Sample {
    /// The relative difference between our wall clock reading and that of the remote clock source.
    clock_offset: i64,
    one_way_delay: u64,
}

#[derive(Clone, Debug)]
struct Epoch {
    /// The best clock offset sample per remote clock source (with minimum one way delay) collected
    /// over the course of a window period of several seconds.
    sources: Vec<Option<Sample>>,

    /// The total number of samples learned while synchronizing this epoch.
    samples: usize,

    /// The monotonic clock timestamp when this epoch began. We use this to measure elapsed time.
    monotonic: u64,

    /// The wall clock timestamp when this epoch began. We add the elapsed monotonic time to this
    /// plus the synchronized clock offset to arrive at a synchronized realtime timestamp. We
    /// capture this realtime when starting the epoch, before we take any samples, to guard against
    /// any jumps in the system's realtime clock from impacting our measurements.
    realtime: i64,

    /// Once we have enough source clock offset samples in agreement, the epoch is synchronized.
    /// We then have lower and upper bounds on the true cluster time, and can install this epoch
    /// for subsequent clock readings. This epoch is then valid for several seconds, while clock
    /// drift has not had enough time to accumulate into any significant clock skew, and while we
    /// collect samples for the next epoch to refresh and replace this one.
    synchronized: Option<Interval>,

    /// A guard to prevent synchronizing too often without having learned any new samples.
    learned: bool,
}

impl Epoch {
    fn new(replica_count: usize) -> Self {
        Self {
            sources: vec![None; replica_count],
            samples: 0,
            monotonic: 0,
            realtime: 0,
            synchronized: None,
            learned: false,
        }
    }

    fn reset(&mut self, replica: u8, monotonic_ns: u64, realtime_ns: i64) {
        self.sources.fill(None);
        // A replica always has zero clock offset and network delay to its own system time reading:
        self.sources[usize::from(replica)] = Some(Sample { clock_offset: 0, one_way_delay: 0 });
        self.samples = 1;
        self.monotonic = monotonic_ns;
        self.realtime = realtime_ns;
        self.synchronized = None;
        self.learned = false;
    }

    fn sources_sampled(&self) -> usize {
        self.sources.iter().filter(|sampled| sampled.is_some()).count()
    }
}

/// Port of `Clock`: provides cluster-synchronized wall-clock time.
pub struct Clock {
    /// The index of the replica using this clock to provide synchronized time.
    replica: u8,
    /// Minimal number of distinct clock sources required for synchronization.
    quorum: u8,

    /// The underlying time source for this clock (system time or deterministic time).
    ///
    /// DEVIATION: upstream stores a vtable-based `Time` value; here a boxed trait object serves
    /// the same role without `unsafe`.
    time: Box<dyn Time>,

    /// An epoch from which the clock can read synchronized clock timestamps within safe bounds.
    /// At least `constants.clock_synchronization_window_min` is needed for this to be ready to use.
    epoch: Epoch,

    /// The next epoch (collecting samples and being synchronized) to replace the current epoch.
    window: Epoch,

    /// A static allocation to convert window samples into tuple bounds for Marzullo's algorithm.
    marzullo_tuples: Vec<Tuple>,

    /// A kill switch to revert to unsynchronized realtime.
    synchronization_disabled: bool,
}

/// Options for [`Clock::new`] (upstream anonymous options struct).
#[derive(Clone, Copy, Debug)]
pub struct ClockOptions {
    /// The size of the cluster, i.e. the number of clock sources (including this replica).
    pub replica_count: u8,
    pub replica: u8,
    pub quorum: u8,
}

impl Clock {
    /// # Panics
    /// Panics if the options violate cluster-shape invariants (`replica < replica_count`,
    /// `quorum <= replica_count`, quorum > 1 for clusters larger than one), mirroring upstream's
    /// assertions.
    #[must_use]
    pub fn new(time: Box<dyn Time>, options: ClockOptions) -> Self {
        assert!(options.replica_count > 0);
        assert!(options.replica < options.replica_count);
        assert!(options.quorum > 0);
        assert!(options.quorum <= options.replica_count);
        if options.replica_count > 1 {
            assert!(options.quorum > 1);
        }

        let replica_count = usize::from(options.replica_count);
        // There are two Marzullo tuple bounds (lower and upper) per source clock offset sample:
        let mut clock = Self {
            replica: options.replica,
            quorum: options.quorum,
            time,
            epoch: Epoch::new(replica_count),
            window: Epoch::new(replica_count),
            marzullo_tuples: vec![
                Tuple { source: 0, offset: 0, bound: Bound::Lower };
                usize::from(options.replica_count) * 2
            ],
            // A cluster of one cannot synchronize.
            synchronization_disabled: options.replica_count == 1,
        };

        let monotonic_ns = clock.monotonic().ns;
        let realtime_ns = clock.realtime();

        // Reset the current epoch to be unsynchronized...
        clock.epoch.reset(clock.replica, monotonic_ns, realtime_ns);
        // ...and open a new epoch window to start collecting samples:
        clock.window.reset(clock.replica, monotonic_ns, realtime_ns);

        clock
    }

    /// Called by `Replica.on_pong()` with:
    /// * the index of the `replica` that has replied to our ping with a pong,
    /// * our monotonic timestamp `m0` embedded in the ping we sent, carried over into this pong,
    /// * the remote replica's `realtime()` timestamp `t1`, and
    /// * our monotonic timestamp `m2` as captured by our `Replica.on_pong()` handler.
    /// # Panics
    /// Panics if `replica == self.replica` or if monotonicity guarantees are violated
    /// (`m2 < window.monotonic`), mirroring upstream's assertions.
    pub fn learn(&mut self, replica: u8, m0: u64, t1: i64, m2: u64) {
        assert_ne!(replica, self.replica);

        if self.synchronization_disabled {
            return;
        }

        // Our m0 and m2 readings should always be monotonically increasing if not equal.
        // Crucially, it is possible for a very fast network to have m0 == m2, especially where
        // `constants.tick_ms` is at a more course granularity. We must therefore tolerate RTT=0 or
        // otherwise we would have a liveness bug simply because we would be throwing away perfectly
        // good clock samples.
        // This condition should never be true. Reject this as a bad sample:
        if m0 > m2 {
            return;
        }

        // The window was reset between a ping and the corresponding pong.
        if m0 < self.window.monotonic {
            return;
        }
        assert!(m2 >= self.window.monotonic); // Guaranteed by monotonicity of our local Time.

        let elapsed: u64 = m2 - self.window.monotonic;
        if elapsed > WINDOW_MAX {
            return;
        }

        let round_trip_time: u64 = m2 - m0;
        let one_way_delay: u64 = round_trip_time / 2;
        let t2: i64 = self.window.realtime.wrapping_add(i64::try_from(elapsed).unwrap_or(i64::MAX));
        let clock_offset: i64 =
            t1.wrapping_add(i64::try_from(one_way_delay).unwrap_or(i64::MAX)).wrapping_sub(t2);
        let asymmetric_delay = self.estimate_asymmetric_delay(replica, one_way_delay, clock_offset);
        let clock_offset_corrected = clock_offset.wrapping_add(asymmetric_delay);

        // The less network delay, the more likely we have an accurate clock offset measurement:
        let candidate = Sample { clock_offset: clock_offset_corrected, one_way_delay };
        self.window.sources[usize::from(replica)] =
            minimum_one_way_delay(self.window.sources[usize::from(replica)], Some(candidate));

        self.window.samples += 1;

        // We decouple calls to `synchronize()` so that it's not triggered by these network events.
        // Otherwise, excessive duplicate network packets would burn the CPU.
        self.window.learned = true;
    }

    /// Called by `Replica.on_ping_timeout()` to provide `m0` when we decide to send a ping.
    /// Called by `Replica.on_pong()` to provide `m2` when we receive a pong.
    /// Called by `Replica.on_commit_message_timeout()` to allow backups to discard
    /// duplicate/misdirected heartbeats.
    pub fn monotonic(&mut self) -> Instant {
        self.time.monotonic()
    }

    /// Called by `Replica.on_ping()` when responding to a ping with a pong.
    /// This should never be used by the state machine, only for measuring clock offsets.
    #[must_use]
    pub fn realtime(&self) -> i64 {
        self.time.realtime()
    }

    /// Called by `Replica.on_request()` when the primary wants to timestamp a batch. If the
    /// primary's clock is not synchronized with the cluster, it must wait until it is.
    /// Returns the system time clamped to be within our synchronized lower and upper bounds.
    /// This is complementary to NTP and allows clusters with very accurate time to make use of it,
    /// while providing guard rails for when NTP is partitioned or unable to correct quickly enough.
    pub fn realtime_synchronized(&mut self) -> Option<i64> {
        if self.synchronization_disabled {
            Some(self.realtime())
        } else if let Some(interval) = self.epoch.synchronized {
            let now_ns = self.monotonic().ns;
            let elapsed_ns = now_ns - self.epoch.monotonic;
            #[allow(clippy::cast_possible_wrap)] // elapsed < epoch_max fits i64 comfortably
            let elapsed = elapsed_ns as i64;
            Some(i64::clamp(
                self.realtime(),
                self.epoch.realtime + elapsed + interval.lower_bound,
                self.epoch.realtime + elapsed + interval.upper_bound,
            ))
        } else {
            None
        }
    }

    #[must_use]
    pub fn round_trip_time_median_ns(&self) -> Option<u64> {
        // +1 to allow for the standby.
        let mut one_way_delays = [0u64; constants::REPLICAS_MAX + 1];
        let mut count = 0;
        for (replica_index, sampled) in self.window.sources.iter().enumerate() {
            if usize::from(self.replica) != replica_index
                && let Some(sample) = sampled
            {
                one_way_delays[count] = sample.one_way_delay;
                count += 1;
            }
        }

        if count < usize::from(self.quorum) {
            None
        } else {
            one_way_delays[..count].sort_unstable();
            let median = one_way_delays[count / 2];
            Some(median * 2)
        }
    }

    pub fn tick(&mut self) {
        self.time.tick();

        if self.synchronization_disabled {
            return;
        }
        self.synchronize();

        // Expire the current epoch if successive windows failed to synchronize:
        // Gradual clock drift prevents us from using an epoch for more than a few seconds.
        let now_ns = self.monotonic().ns;
        if now_ns - self.epoch.monotonic >= EPOCH_MAX {
            self.epoch.reset(self.replica, now_ns, self.realtime());
        }
    }

    /// Estimates the asymmetric delay for a sample compared to the previous window, according to
    /// Algorithm 1 from Section 4.2,
    /// "A System for Clock Synchronization in an Internet of Things".
    ///
    /// Note that it is impossible to estimate persistent asymmetric delay, as these two situations
    /// are indistinguishable:
    /// - A and B have synchronized clocks and a 50ms symmetrical delay.
    /// - B's clock is 50ms ahead, A → B delay is 0ms, B → A delay is 100ms.
    ///
    /// In both of these cases, A and B observe that a ping-pong round trip takes 100ms and that
    /// a pong's timestamp is 50ms ahead of ping's timestamp.
    ///
    /// Instead, the model here is of a one-time delay --- a particular ping or pong message got
    /// delayed because it had a large prepare message in front of it in the send queue, a network
    /// packet got lost, or a pigeon got eaten by a cat.
    ///
    /// The delay happened either for the ping (forward path) or for the pong (reverse path)
    /// message. Assuming that the minimum RTT seen before is a no-delay situation, the magnitude of
    /// a delay for the current sample can be estimated as RTT - min(RTT), and the direction
    /// (forward/reverse) distinguished by comparing unadjusted clock offsets.
    ///
    /// Previous window is used to determine min(RTT).
    fn estimate_asymmetric_delay(&self, replica: u8, one_way_delay: u64, clock_offset: i64) -> i64 {
        // Note that `one_way_delay` may be 0 for very fast networks.

        // 10 * std.time.ns_per_ms:
        const ERROR_MARGIN: i64 = 10_000_000;
        let error_margin = ERROR_MARGIN;

        if let Some(epoch_sample) = self.epoch.sources[usize::from(replica)] {
            if one_way_delay <= epoch_sample.one_way_delay {
                0
            } else if clock_offset > epoch_sample.clock_offset + error_margin {
                // The asymmetric error is on the forward network path.
                let delta = one_way_delay - epoch_sample.one_way_delay;
                0 - i64::try_from(delta).unwrap_or(i64::MAX)
            } else if clock_offset < epoch_sample.clock_offset - error_margin {
                // The asymmetric error is on the reverse network path.
                let delta = one_way_delay - epoch_sample.one_way_delay;
                i64::try_from(delta).unwrap_or(i64::MAX)
            } else {
                0
            }
        } else {
            0
        }
    }

    fn synchronize(&mut self) {
        assert!(self.window.synchronized.is_none());

        // Wait until the window has enough accurate samples:
        let now_ns = self.monotonic().ns;
        let elapsed = now_ns - self.window.monotonic;
        if elapsed < WINDOW_MIN {
            return;
        }
        if elapsed >= WINDOW_MAX {
            // We took too long to synchronize the window, expire stale samples...
            let sources_sampled = self.window.sources_sampled();
            let _ = sources_sampled; // logged upstream; see module DEVIATION on logging
            self.window.reset(self.replica, now_ns, self.realtime());
            return;
        }

        if !self.window.learned {
            return;
        }
        // Do not reset `learned` any earlier than this (before we have attempted to synchronize).
        self.window.learned = false;

        assert!(
            self.window.sources[usize::from(self.replica)]
                .is_some_and(|s| s.clock_offset == 0 && s.one_way_delay == 0)
        );

        // Starting with the most clock offset tolerance, while we have a quorum, find the best
        // smallest interval with the least clock offset tolerance, reducing tolerance at each step:
        let mut tolerance: u64 = CLOCK_OFFSET_TOLERANCE_MAX;
        let mut terminate = false;
        let mut rounds: usize = 0;
        // Do at least one round if tolerance=0 and cap the number of rounds to avoid runaway loops.
        loop {
            if terminate || rounds >= 64 {
                break;
            }
            if tolerance == 0 {
                terminate = true;
            }
            rounds += 1;

            let interval = {
                // Lend the scratch buffer to Marzullo for this round:
                let mut tuples = std::mem::take(&mut self.marzullo_tuples);
                fill_window_tuples(&mut tuples, &self.window.sources, tolerance);
                let result = Marzullo::smallest_interval(&mut tuples);
                self.marzullo_tuples = tuples;
                result
            };
            if u16::from(interval.sources_true) < u16::from(self.quorum) {
                break;
            }

            // The new interval may reduce the number of `sources_true` while also decreasing error.
            // In other words, provided we maintain a quorum, we prefer tighter tolerance bounds.
            self.window.synchronized = Some(interval);

            tolerance /= 2;
        }

        // Wait for more accurate samples or until we timeout the window for lack of quorum:
        if self.window.synchronized.is_none() {
            return;
        }

        // Transitioning from not being synchronized to being synchronized - logged upstream;
        // see module DEVIATION on logging.

        std::mem::swap(&mut self.epoch, &mut self.window);
        let now_ns = self.monotonic().ns;
        let realtime_ns = self.realtime();
        self.window.reset(self.replica, now_ns, realtime_ns);

        self.after_synchronization(now_ns, realtime_ns);
    }

    fn after_synchronization(&mut self, now_ns: u64, system_realtime: i64) {
        let Some(new_interval) = self.epoch.synchronized else {
            unreachable!("after_synchronization requires a synchronized epoch");
        };

        let elapsed_ns = now_ns - self.epoch.monotonic;
        #[allow(clippy::cast_possible_wrap)] // bounded by epoch_max, far below i64 range
        let elapsed = elapsed_ns as i64;
        let lower = self.epoch.realtime + elapsed + new_interval.lower_bound;
        let upper = self.epoch.realtime + elapsed + new_interval.upper_bound;
        let cluster = i64::clamp(system_realtime, lower, upper);

        // The only current hard limit on what the clock skew can actually be is from
        // `clock_offset_tolerance_max`.
        //
        // Warn at 50ms, since that's a reasonable amount of NTP clock skew, and ensure that 50ms
        // is a reasonable (sub 1%) portion of `clock_offset_tolerance_max`.
        // (Logging/trace gauges omitted; see module DEVIATION.)
        //
        // The only externally visible effect of this function is the clamp performed by
        // `realtime_synchronized`; these values exist upstream solely for the omitted log line.
        let _ = (system_realtime == cluster, lower, upper);
    }
}

/// Builds Marzullo tuples from the window's samples within `tolerance`; returns tuple count.
///
/// DEVIATION: upstream fills `self.marzullo_tuples` and returns a subslice; here the fill is
/// factored into a free function so the borrow checker can see that the scratch buffer and the
/// window samples don't alias. Same contents, same order.
#[allow(clippy::cast_possible_wrap)] // one_way_delay + tolerance are small relative to i64 range
fn fill_window_tuples(tuples: &mut [Tuple], sources: &[Option<Sample>], tolerance: u64) -> usize {
    let mut count = 0;
    for (source, sampled) in sources.iter().enumerate() {
        if let Some(sample) = sampled {
            #[allow(clippy::cast_possible_truncation)] // source < replica_count <= u8::MAX
            let source = source as u8;
            let padding = i64::try_from(sample.one_way_delay + tolerance).unwrap_or(i64::MAX);
            tuples[count] =
                Tuple { source, offset: sample.clock_offset - padding, bound: Bound::Lower };
            count += 1;
            tuples[count] =
                Tuple { source, offset: sample.clock_offset + padding, bound: Bound::Upper };
            count += 1;
        }
    }
    count
}

fn minimum_one_way_delay(a: Option<Sample>, b: Option<Sample>) -> Option<Sample> {
    match (a, b) {
        (None, _) => b,
        (_, None) => a,
        (Some(a), Some(b)) => {
            if a.one_way_delay < b.one_way_delay {
                Some(a)
            } else {
                // Choose B if B's one way delay is less or the same (we assume B is newer):
                Some(b)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Test scaffolding mirrors upstream's @intCast-heavy simulation math; every value is bounded
    // by the simulation parameters, so the narrowing casts are intentional.
    #![allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::redundant_closure_for_method_calls
    )]

    use super::*;
    use crate::testing::time::{OffsetType, SharedTimeSim, TimeSim};
    use tigerbeetle_core::stdx::{NS_PER_DAY, NS_PER_MS, NS_PER_S};

    /// Port of `ClockUnitTestContainer`: drives a 3-replica clock with two identical remote
    /// sources reporting samples over a simulated network.
    struct ClockUnitTestContainer {
        time: SharedTimeSim,
        clock: Clock,
        rtt: u64,
        owd: u64,
        learn_interval: u64,
    }

    impl ClockUnitTestContainer {
        fn new(
            offset_type: OffsetType,
            offset_coefficient_a: i64,
            offset_coefficient_b: i64,
        ) -> Self {
            let time = SharedTimeSim::new(TimeSim::new(
                NS_PER_S / 2, // resolution
                offset_type,
                offset_coefficient_a,
                offset_coefficient_b,
            ));
            let clock = Clock::new(
                Box::new(time.clone()),
                ClockOptions { replica_count: 3, replica: 0, quorum: 2 },
            );
            Self { time, clock, rtt: 300 * NS_PER_MS, owd: 150 * NS_PER_MS, learn_interval: 5 }
        }

        fn run_till_tick(&mut self, tick_stop: u64) {
            while self.time.with(|t| t.ticks) < tick_stop {
                self.time.with_mut(|t| t.tick());

                if self.time.with(|t| t.ticks % self.learn_interval) == 0 {
                    let on_pong_time = self.clock.monotonic().ns;
                    let m0 = on_pong_time - self.rtt;
                    let t1 = on_pong_time as i64 - self.owd as i64;

                    self.clock.learn(1, m0, t1, on_pong_time);
                    self.clock.learn(2, m0, t1, on_pong_time);
                }

                self.clock.synchronize();
            }
        }

        /// The synchronized clock reading, or panic (tests upstream use `.?`).
        fn realtime_synchronized(&mut self) -> i64 {
            match self.clock.realtime_synchronized() {
                Some(realtime) => realtime,
                None => panic!("clock not synchronized"),
            }
        }

        /// Expected `(tick, expected_offset)` pairs; port of `ticks_to_perform_assertions`.
        #[allow(clippy::cast_possible_wrap)] // simulation parameters are small
        fn ticks_to_perform_assertions(&self) -> [(u64, i64); 3] {
            match self
                .time
                .with(|t| (t.offset_type, t.offset_coefficient_a, t.offset_coefficient_b))
            {
                (OffsetType::Linear, a, _b) => {
                    // For the first (OWD/drift per tick) ticks, the offset < OWD. This means that
                    // the Marzullo interval is [0,0] (the offset and OWD are 0 for a replica
                    // w.r.t. itself). Therefore the offset of `clock.realtime_synchronized` will
                    // be the analytically prescribed offset at the start of the window.
                    // Beyond this, the offset > OWD and the Marzullo interval will be from replica
                    // 1 and replica 2. The `clock.realtime_synchronized` will be clamped to the
                    // lower bound. Therefore the `clock.realtime_synchronized` will be offset by
                    // the OWD.
                    let threshold = self.owd / a as u64;
                    [
                        (threshold, self.time.with(|t| t.offset(threshold - self.learn_interval))),
                        (threshold + 100, self.owd as i64),
                        (threshold + 200, self.owd as i64),
                    ]
                }
                (OffsetType::Periodic, _a, b) => {
                    let b = b as u64;
                    [(b / 4, self.owd as i64), (b / 2, 0), (b * 3 / 4, -(self.owd as i64))]
                }
                (OffsetType::Step, _a, b) => {
                    let b = b as u64;
                    [(b - 10, 0), (b + 10, -(self.owd as i64)), (b + 10, -(self.owd as i64))]
                }
            }
        }
    }

    /// Upstream: test "ideal clocks get clamped to cluster time".
    #[test]
    fn ideal_clocks_get_clamped_to_cluster_time() {
        // Linear drift clock that loses 1ms per tick.
        let mut linear = ClockUnitTestContainer::new(OffsetType::Linear, NS_PER_MS as i64, 0);
        for (tick, expected_offset) in linear.ticks_to_perform_assertions() {
            linear.run_till_tick(tick);
            let offset = linear.clock.monotonic().ns as i64 - linear.realtime_synchronized();
            assert_eq!(offset, expected_offset, "linear @ tick {tick}");
        }

        // Periodic drift clock that loses up to 1s with a period of 200 ticks.
        let mut periodic = ClockUnitTestContainer::new(OffsetType::Periodic, NS_PER_S as i64, 200);
        for (tick, expected_offset) in periodic.ticks_to_perform_assertions() {
            periodic.run_till_tick(tick);
            let offset = periodic.clock.monotonic().ns as i64 - periodic.realtime_synchronized();
            assert_eq!(offset, expected_offset, "periodic @ tick {tick}");
        }

        // Jumping clock that jumps 5 days ahead after 49 ticks.
        let mut step = ClockUnitTestContainer::new(OffsetType::Step, -5 * NS_PER_DAY as i64, 49);
        for (tick, expected_offset) in step.ticks_to_perform_assertions() {
            step.run_till_tick(tick);
            let offset = step.clock.monotonic().ns as i64 - step.realtime_synchronized();
            assert_eq!(offset, expected_offset, "step @ tick {tick}");
        }
    }

    /// Sanity check of the median round trip time helper: feed two identical samples directly.
    #[test]
    fn round_trip_time_median() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let shared_time = time.clone();
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });
        assert!(clock.round_trip_time_median_ns().is_none());

        // One tick (500ms at this resolution) so that `now` exceeds the RTT:
        shared_time.with_mut(|t| t.tick());
        let now = clock.monotonic().ns;
        let t1 = now as i64 - 150 * NS_PER_MS as i64;
        clock.learn(1, now - 300 * NS_PER_MS, t1, now);
        clock.learn(2, now - 300 * NS_PER_MS, t1, now);

        // Both remotes report the same constant RTT (300ms), so the median is exactly 300ms:
        assert_eq!(clock.round_trip_time_median_ns(), Some(300 * NS_PER_MS));
    }

    #[test]
    fn new_rejects_invalid_options() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let invalid = [
            ClockOptions { replica_count: 0, replica: 0, quorum: 0 },
            ClockOptions { replica_count: 3, replica: 3, quorum: 2 },
            ClockOptions { replica_count: 3, replica: 0, quorum: 0 },
            ClockOptions { replica_count: 3, replica: 0, quorum: 4 },
            ClockOptions { replica_count: 3, replica: 0, quorum: 1 },
        ];
        for options in invalid {
            let t = time.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = Clock::new(Box::new(t), options);
            }));
            assert!(result.is_err(), "invalid clock options must panic");
        }

        // A single-replica cluster is legal and disables synchronization.
        let clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 1, replica: 0, quorum: 1 });
        assert!(clock.synchronization_disabled);
    }

    #[test]
    fn learn_rejects_bad_samples() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let shared_time = time.clone();
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });
        let baseline_samples = clock.window.samples;

        // A reading that goes backwards (m0 > m2) is rejected outright.
        clock.learn(1, 10, 0, 5);
        assert_eq!(clock.window.samples, baseline_samples);
        assert!(clock.window.sources[1].is_none());

        // Learning from ourselves is an invariant violation.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clock.learn(0, 5, 0, 10);
        }));
        assert!(result.is_err(), "learning from self must panic");

        // A sample that spans more than the synchronization window max is too stale.
        shared_time.with_mut(|t| t.tick());
        let now = clock.monotonic().ns;
        clock.learn(1, now - NS_PER_MS, now as i64, now + 2 * WINDOW_MAX);
        assert_eq!(clock.window.samples, baseline_samples);
        assert!(clock.window.sources[1].is_none());
    }

    #[test]
    fn learn_rejects_sample_from_before_window_reset() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let shared_time = time.clone();
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });

        // 41 ticks × 500ms = 20.5s, past `WINDOW_MAX`: the next synchronize() expires the window
        // and resets its monotonic start, so no prior ping can count towards it.
        for _ in 0..41 {
            shared_time.with_mut(|t| t.tick());
        }
        let expired_window_monotonic = clock.window.monotonic;
        clock.synchronize();
        assert!(clock.window.monotonic > expired_window_monotonic);

        // A pong whose ping predates the window reset is rejected (m0 < window.monotonic).
        let m2 = clock.monotonic().ns;
        let m0 = m2 - NS_PER_S;
        clock.learn(1, m0, (m2 - 2 * NS_PER_MS) as i64, m2);
        assert_eq!(clock.window.samples, 1);
        assert!(clock.window.sources[1].is_none());
    }

    #[test]
    fn realtime_synchronized_disabled_and_unsynchronized() {
        // Single-replica cluster: synchronization is disabled and realtime is served directly.
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 1, replica: 0, quorum: 1 });
        assert!(clock.synchronization_disabled);
        assert_eq!(clock.realtime_synchronized(), Some(clock.realtime()));

        // Multi-replica cluster before the first successful synchronization: None.
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });
        assert!(!clock.synchronization_disabled);
        assert!(clock.realtime_synchronized().is_none());
    }

    #[test]
    fn tick_disabled_cluster_advances_time_only() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 1, replica: 0, quorum: 1 });
        let before = clock.monotonic().ns;
        clock.tick();
        assert_eq!(clock.monotonic().ns, before + NS_PER_S / 2);
        assert!(clock.window.synchronized.is_none());
    }

    #[test]
    fn minimum_one_way_delay_keeps_fastest_or_newest() {
        let fast = Some(Sample { clock_offset: 10, one_way_delay: 100 });
        let slow = Some(Sample { clock_offset: 20, one_way_delay: 300 });

        // Empty slots forward the existing sample of the other kind.
        let keep = minimum_one_way_delay(None, slow);
        assert!(keep.is_some_and(|s| s.clock_offset == 20 && s.one_way_delay == 300));
        let keep = minimum_one_way_delay(slow, None);
        assert!(keep.is_some_and(|s| s.clock_offset == 20 && s.one_way_delay == 300));

        // The slower existing sample is replaced by the faster one.
        let keep = minimum_one_way_delay(slow, fast);
        assert!(keep.is_some_and(|s| s.clock_offset == 10 && s.one_way_delay == 100));

        // A tie keeps the incoming (newer, higher offset) sample.
        let tie_new = Some(Sample { clock_offset: 30, one_way_delay: 100 });
        let keep = minimum_one_way_delay(fast, tie_new);
        assert!(keep.is_some_and(|s| s.clock_offset == 30 && s.one_way_delay == 100));

        // The faster existing sample is retained over the slower incoming one.
        let keep = minimum_one_way_delay(fast, slow);
        assert!(keep.is_some_and(|s| s.clock_offset == 10 && s.one_way_delay == 100));
    }

    #[test]
    fn estimate_asymmetric_delay_identifies_forward_and_reverse_delays() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });

        // No prior epoch sample for this replica: no correction available.
        assert_eq!(clock.estimate_asymmetric_delay(1, 200 * NS_PER_MS, 0), 0);

        // Seed the epoch with a baseline (self-consistent, 100ms one-way delay).
        clock.epoch.sources[1] = Some(Sample { clock_offset: 0, one_way_delay: 100 * NS_PER_MS });

        // Not slower than the baseline: no correction.
        assert_eq!(clock.estimate_asymmetric_delay(1, 80 * NS_PER_MS, 0), 0);
        assert_eq!(clock.estimate_asymmetric_delay(1, 100 * NS_PER_MS, 0), 0);

        // Slower and offset still within the error margin: no correction.
        assert_eq!(clock.estimate_asymmetric_delay(1, 300 * NS_PER_MS, 5 * NS_PER_MS as i64), 0);

        // Slower and offset ahead of baseline: the extra delay is on the forward path.
        assert_eq!(
            clock.estimate_asymmetric_delay(1, 300 * NS_PER_MS, 200 * NS_PER_MS as i64),
            -(200 * NS_PER_MS as i64)
        );

        // Slower and offset behind baseline: the extra delay is on the reverse path.
        assert_eq!(
            clock.estimate_asymmetric_delay(1, 300 * NS_PER_MS, -(200 * NS_PER_MS as i64)),
            200 * NS_PER_MS as i64
        );

        // A replica with no baseline keeps correcting from its own epoch sample: source 2 has no
        // sample, so a wild offset yields no correction.
        assert_eq!(clock.estimate_asymmetric_delay(2, 300 * NS_PER_MS, 200 * NS_PER_MS as i64), 0);
    }

    #[test]
    fn round_trip_time_median_detail() {
        let time =
            SharedTimeSim::new(TimeSim::new(NS_PER_S / 2, OffsetType::Linear, NS_PER_MS as i64, 0));
        let shared_time = time.clone();
        let mut clock =
            Clock::new(Box::new(time), ClockOptions { replica_count: 3, replica: 0, quorum: 2 });

        shared_time.with_mut(|t| t.tick());
        let now = clock.monotonic().ns;

        // A single remote source is below the quorum of 2.
        clock.learn(1, now - 100 * NS_PER_MS, now as i64 - 50 * NS_PER_MS as i64, now);
        assert_eq!(clock.round_trip_time_median_ns(), None);

        // A second source at 300ms RTT: median over {50ms, 150ms} one-way → 300ms RTT.
        clock.learn(2, now - 300 * NS_PER_MS, now as i64 - 150 * NS_PER_MS as i64, now);
        assert_eq!(clock.round_trip_time_median_ns(), Some(300 * NS_PER_MS));

        // Relearning replica 2 at a lower RTT (200ms) refines its one-way delay downward.
        clock.learn(2, now - 200 * NS_PER_MS, now as i64 - 100 * NS_PER_MS as i64, now);
        assert_eq!(clock.round_trip_time_median_ns(), Some(200 * NS_PER_MS));

        // A higher RTT for the same replica is ignored (minimum one-way delay retained).
        clock.learn(2, now - 400 * NS_PER_MS, now as i64 - 200 * NS_PER_MS as i64, now);
        assert_eq!(clock.round_trip_time_median_ns(), Some(200 * NS_PER_MS));
    }
}
