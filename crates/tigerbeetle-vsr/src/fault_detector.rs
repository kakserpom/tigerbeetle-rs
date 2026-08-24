//! FaultDetector estimates the probability that the primary crashed.
//! It is implemented as a pure algorithm in a "sans-io" style.
//!
//! Intuition: imagine looking from a window at the street below and trying to guess whether the
//! nearest traffic light (not directly visible) is red or green. If you see the cars going, it
//! definitely was green recently. If there are no cars, it could be that the light is red, or that
//! you are out of peak hour and there are no cars at all. So, a reasonable algorithm is to note
//! the average interarrival time over a sliding window, and signal "red" when there's a suspicious
//! absence of cars, relative to recent history. Note latency-throughput interaction: if the cars
//! are frequent, but there's a large distance between your window and the traffic light, you notice
//! red very quickly, but only after "wave edge" reaches you.
//!
//! Applying to TigerBeetle, let's consider the case where the cluster is under stable load and
//! prepares are flowing regularly. In this case, a backup uses a sliding window to compute current
//! rate of prepares, and sounds an alarm (ExitView) if the flow stops.
//!
//! To solve the case of an idle cluster, primary broadcasts Commit messages periodically, which are
//! interchangeable with Prepare messages as a signal that the primary is alive.
//!
//! Another complication is that prepare load can be bursty. If the load ramps up quickly, the
//! primary is seen as extremely alive, which is OK. However, if the load is cut off exogenously
//! (because the client finished a batch job, not because the primary crashed), this will look like
//! a dead primary. To solve this, primary uses central-bank style algorithm, where it injects extra
//! Commit messages to keep prepare rate relatively stable in the short run.
//!
//! Upstream: `src/vsr/fault_detector.zig`.

use tigerbeetle_core::stdx::{Duration, Instant, NS_PER_HOUR};

/// Whether the signal is overdue:
///  * [`Tardy::Green`] --- signal is on time
///  * [`Tardy::Yellow`] --- signal seems delayed/lost
///  * [`Tardy::Red`] --- signaler is likely dead
///
/// On yellow, the primary injects a Commit.
/// On red, a backup sends ExitView.
///
/// Rough model:
/// - Random delays, but 2X delay is suspicious.
/// - An individual signal can get lost.
///
///   Either of the above suggests 2X+some as a cutoff point for a fault. We round up to 3X
///   cutoff for red, and 1.5X cutoff for yellow.
///
/// An alternative approach would be to build a probabilistic model of the prepare/commit arrival
/// process, and then compute the actual probability of primary being dead using Bayes' rule. We
/// don't do that, because we don't know the actual underlying model. A simple rule like the above
/// will not give us the optimal answer, but it should work in variety of different contexts!
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tardy {
    Green,
    Yellow,
    Red,
}

#[derive(Clone, Copy, Debug)]
pub struct FaultDetector {
    interval_min: Duration,
    interval_max: Duration,

    signal_last: Instant,
    interval_ewma: Duration,
}

#[derive(Clone, Copy)]
pub struct FaultDetectorOptions {
    pub now: Instant,
    pub interval_min: Duration,
    pub interval_max: Duration,
}

impl FaultDetector {
    /// # Panics
    /// Panics unless `interval_min < interval_max <= 10 hours` (upstream asserts; the upper
    /// bound doubles as overflow protection for the ewma).
    #[must_use]
    pub fn new(options: FaultDetectorOptions) -> Self {
        assert!(options.interval_min.ns < options.interval_max.ns);
        // Sanity check and overflow protection for ewma.
        assert!(options.interval_max.ns <= 10 * NS_PER_HOUR);

        Self {
            interval_min: options.interval_min,
            interval_max: options.interval_max,
            signal_last: options.now,
            interval_ewma: options.interval_max,
        }
    }

    /// # Panics
    /// Panics if `now` precedes the last signal (upstream asserts).
    pub fn signal(&mut self, now: Instant) {
        let past = self.signal_last;
        assert!(past.ns <= now.ns);
        // Clamp first, then ewma_add, to avoid overflows.
        let elapsed = past.elapsed(now).clamp(self.interval_min, self.interval_max);

        self.interval_ewma = ewma_add_duration(self.interval_ewma, elapsed);
        self.signal_last = now;
    }

    /// Is the signal overdue? See [`Tardy`].
    ///
    /// # Panics
    /// Panics if `now` precedes the last signal (upstream asserts).
    #[must_use]
    pub fn tardy(&self, now: Instant) -> Tardy {
        let past = self.signal_last;
        assert!(past.ns <= now.ns);
        let elapsed = past.elapsed(now);

        if elapsed.ns.saturating_mul(2) <= self.interval_ewma.ns * 3 {
            // interval <= 1.5 * interval_ewma
            return Tardy::Green;
        }
        assert!(elapsed.ns >= self.interval_ewma.ns);
        if elapsed.ns <= self.interval_ewma.ns * 3 {
            return Tardy::Yellow;
        }
        assert!(elapsed.ns > self.interval_ewma.ns);
        Tardy::Red
    }

    /// # Panics
    /// Panics if `now` precedes the last signal (upstream asserts).
    pub fn reset(&mut self, now: Instant) {
        let past = self.signal_last;
        assert!(past.ns <= now.ns);
        *self = Self::new(FaultDetectorOptions {
            now,
            interval_min: self.interval_min,
            interval_max: self.interval_max,
        });
    }
}

#[must_use]
fn ewma_add_duration(old: Duration, new: Duration) -> Duration {
    Duration { ns: (old.ns * 4 + new.ns) / 5 }
}

#[cfg(test)]
mod tests {
    use super::{FaultDetector, FaultDetectorOptions, Tardy};
    use tigerbeetle_core::stdx::{Duration, Instant};

    #[test]
    fn fault_detector_smoke() {
        // Test that computed ewma interval tracks actual interval,
        // clamped to limits.
        let mut now = Instant { ns: 1_000 };
        let mut detector = FaultDetector::new(FaultDetectorOptions {
            now,
            interval_min: Duration::ms(100),
            interval_max: Duration::ms(2_000),
        });
        assert_eq!(detector.tardy(now), Tardy::Green);
        assert_eq!(detector.interval_ewma.to_ms(), 2_000);

        for _ in 0..100 {
            now = now.add(Duration::ms(200));
            detector.signal(now);
        }
        now = now.add(Duration::ms(200));
        assert_eq!(detector.tardy(now), Tardy::Green);
        assert_eq!(detector.interval_ewma.to_ms(), 200);

        now = now.add(Duration::ms(200));
        assert_eq!(detector.tardy(now), Tardy::Yellow);

        now = now.add(Duration::ms(250));
        assert_eq!(detector.tardy(now), Tardy::Red);

        for _ in 0..100 {
            now = now.add(Duration::ms(1_000));
            detector.signal(now);
        }
        now = now.add(Duration::ms(1_000));
        assert_eq!(detector.tardy(now), Tardy::Green);
        assert_eq!(detector.interval_ewma.to_ms(), 999);

        for _ in 0..100 {
            now = now.add(Duration::ms(10));
            detector.signal(now);
        }
        now = now.add(Duration::ms(10));
        assert_eq!(detector.tardy(now), Tardy::Green);
        assert_eq!(detector.interval_ewma.to_ms(), 100);

        for _ in 0..100 {
            now = now.add(Duration::ms(10_000));
            detector.signal(now);
        }
        now = now.add(Duration::ms(10_000));
        assert_eq!(detector.tardy(now), Tardy::Red);
        assert_eq!(detector.interval_ewma.to_ms(), 1_999);
    }

    #[test]
    fn fault_detector_smoothing() {
        // Check that, after a burst of prepares is abruptly ended,
        // the primary can gradually reduce the arrival interval,
        // without triggering a view change.
        let mut now = Instant { ns: 1_000 };
        let mut primary = FaultDetector::new(FaultDetectorOptions {
            now,
            interval_min: Duration::ms(50),
            interval_max: Duration::ms(2_000),
        });

        let backup_delay = Duration::ms(100);
        let mut backup = FaultDetector::new(FaultDetectorOptions {
            now,
            interval_min: Duration::ms(50),
            interval_max: Duration::ms(2_000),
        });

        let commit_interval = Duration::ms(500);
        let request_interval = Duration::ms(100);

        let mut commit_timer = now;
        let mut request_timer = now;

        for _ in 0..1_000 {
            now = now.add(Duration::ms(10)); // Advance by one tick.

            // Primary broadcasts commit message every 500ms.
            if commit_timer.elapsed(now).ns > commit_interval.ns {
                commit_timer = now;
                primary.signal(now);
                backup.signal(now.add(backup_delay));
            }
            assert_eq!(primary.tardy(now), Tardy::Green);

            // Primary converts a request into prepare every 100ms.
            if request_timer.elapsed(now).ns > request_interval.ns {
                request_timer = now;
                primary.signal(now);
                backup.signal(now.add(backup_delay));
            }
            assert_eq!(backup.tardy(now.add(backup_delay)), Tardy::Green);
        }
        assert_eq!(primary.interval_ewma.to_ms(), 95);
        assert_eq!(backup.interval_ewma.to_ms(), 95);

        for _ in 0..1_000 {
            now = now.add(Duration::ms(10)); // Advance by one tick.

            if commit_timer.elapsed(now).ns > commit_interval.ns {
                commit_timer = now;
                primary.signal(now);
                backup.signal(now.add(backup_delay));
            }
            match primary.tardy(now) {
                Tardy::Green => {}
                Tardy::Yellow => {
                    // Stay awhile and listen, wanderer, the story of the next line.
                    //
                    // The original implementation in replica.zig didn't have the equivalent.
                    // In other words, the primary was always sending a commit message every
                    // 500 ms, and then additionally injecting a "bonus" commit whenever fault
                    // detector flashed yellow. That is, when the delay since last
                    // commit/prepare exceeded 1.5X of the current interval. Because 1.5 > 1,
                    // it should be the case that the ewma of the interval gradually converges
                    // to 500 ms, right?
                    //
                    // Wrong! Implementing this test to double-check "obviously correct" logic
                    // showed that the interval expands from 100ms to 250ms, but then gets
                    // stuck! Here's the picture. Originally we start with an idle cluster
                    // where only commits are pulsed periodically:
                    //
                    // C       C       C       C       C
                    //
                    // Then the load starts, and we get a lot of prepares in between:
                    //
                    // C P P P C P P P C P P P C P P P C
                    //
                    // Then, the load cuts off, but the primary starts Injecting extra commits
                    // to keep the interval:
                    //
                    // C I I I C I I   C   I   C ....
                    //
                    // The frequency of injected commits goes down to just one, but then gets
                    // stuck:
                    //
                    // C     I C     I C     I C     I C
                    //
                    // Both the first and the last lines are fixed points. For the first line,
                    // the ewma is 500ms and no commits are injected. For the last line, the
                    // ewma is 250ms, and at 250*1.5=350ms after the last C, an I gets
                    // injected.
                    //
                    // More generally, if we are currently injecting one extra message, we need
                    // to tolerate at least 2X injection delay to get to zero extra messages,
                    // _if_ we keep a steady pace of heartbeat commits. This means that delay
                    // to detect a crashed primary needs to grow proportionally more, and that
                    // feels like too much of a lag. Instead, we let go of the constraint of
                    // keeping the heartbeat steady, and skip the next normal beat whenever we
                    // inject one.
                    commit_timer = now;

                    primary.signal(now);
                    backup.signal(now.add(backup_delay));
                }
                Tardy::Red => unreachable!("primary must never look dead in this test"),
            }
            match backup.tardy(now.add(backup_delay)) {
                Tardy::Green | Tardy::Yellow => {}
                Tardy::Red => unreachable!("backup must never see a dead primary here"),
            }
        }
        assert_eq!(primary.interval_ewma.to_ms(), 499);
        assert_eq!(backup.interval_ewma.to_ms(), 499);
    }
}
