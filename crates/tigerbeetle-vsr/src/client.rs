//! Port of `src/vsr/client.zig` — the client side of the Viewstamped
//! Replication protocol.
//!
//! The client registers a session, runs at most one request inflight at a
//! time, retransmits on timeout with exponential backoff + hedging to a random
//! backup, calibrates its request timeout from ping/pong round-trip times, and
//! handles eviction.
//!
//! DEVIATION (ownership): upstream is `ClientType(StateMachineOperation,
//! MessageBus)` with ref-counted `*Message`s and a pointer-equality assertion
//! (`send_request_for_the_first_time` verifies the inflight pointer). Here the
//! [`Client`] owns a [`MessagePool`], the message is moved into
//! `request_inflight`, and a resend clones the buffer for each destination
//! instead of deferring to the bus's reference count.
//!
//! DEVIATION (bus): upstream embeds a `MessageBus` (with its own pool and
//! lifecycle: `init`/`tick_client`/`shutdown`/`deinit`). Here the outbound
//! transport is the sans-IO [`Bus`] trait; the connection lifecycle is deferred
//! with the TCP layer, and inbound messages are delivered directly by the
//! caller via [`Client::on_message`] instead of upstream `on_messages`, which
//! the bus fires while draining its receive buffers.
//!
//! DEVIATION (callbacks): callbacks are boxed closures instead of bare fn
//! pointers carrying a `user_data` arg; `user_data` remains an explicit
//! parameter for the callback's own use (the log/user_data plumbing collapses
//! into the closure captures).
//!
//! DEVIATION (timer): upstream's `request_completion_timer` (`vsr.time.Timer`)
//! is reset per request but never read in the current upstream source; it is
//! not ported until a consumer exists. TODO(port): src/vsr/client.zig:87.

use tigerbeetle_core::constants;
use tigerbeetle_core::stdx::prng::Prng;
use tigerbeetle_core::types::{Account, AccountFilter, ChangeEventsFilter, QueryFilter, Transfer};

use crate::command::Command;
use crate::message::Message;
use crate::message_header::{
    self, Eviction, PingClient, PongClient, Reply, Request, SIZE_U32, TypedHeader,
};
use crate::message_pool::MessagePool;
use crate::multiversion::Release;
use crate::time::Time;
use crate::{Operation, RegisterRequest, RegisterResult};

/// The client's outbound transport (upstream: `MessageBus`).
///
/// The bus is responsible for copying the message bytes onto its own send
/// queue (upstream bumps the message's reference count per destination, which
/// safe Rust cannot do for a shared buffer — the [`Client`] clones the input
/// per destination).
pub trait Bus {
    /// Queue `message` for delivery to replica `replica`.
    fn send(&mut self, replica: u8, message: &Message);
}

/// A request completion callback (upstream `Request.Callback`).
type RequestCallback = dyn FnMut(u128, Operation, u64, &[u8]);

/// A register completion callback (upstream `Request.RegisterCallback`).
type RegisterCallback = dyn FnMut(u128, &RegisterResult);

/// A callback-bridging enum mirroring upstream `Client.Request.Callback` (a
/// union keyed on whether the inflight message is a register).
pub enum Callback {
    /// `operation != register`: reports the operation, the prepare's timestamp
    /// and the reply body.
    Request(Box<RequestCallback>),
    /// `operation == register`: reports the echoed `RegisterResult`.
    Register(Box<RegisterCallback>),
}

/// The in-flight request (upstream `Client.Request`).
pub struct RequestInflight {
    /// The request message, checksums finalized on first send.
    pub message: Message,
    /// Opaque caller data echoed to the callback.
    pub user_data: u128,
    /// The completion callback (register vs. request).
    pub callback: Callback,
}

/// A reply test hook (upstream `on_reply_callback`).
type ReplyCallback<B> = dyn FnMut(&mut Client<B>, &Message, &Message);

/// An eviction hook (upstream `on_eviction_callback`).
type EvictionCallback<B> = dyn FnMut(&mut Client<B>, &Eviction);

/// Options for [`Client::new`].
pub struct ClientOptions<B: Bus> {
    /// A universally unique identifier for the client (must not be zero).
    pub id: u128,
    /// The identifier for the cluster that this client intends to communicate with.
    pub cluster: u128,
    /// The number of replicas in the cluster.
    pub replica_count: u8,
    /// Whether this client is recovering from an append-only file.
    pub aof_recovery: bool,
    /// Eviction hook. `None` means an eviction panics (upstream's null callback).
    pub eviction_callback: Option<Box<EvictionCallback<B>>>,
}

/// Port of `vsr.Timeout` (`src/vsr.zig:733`).
///
/// A tick-counting timeout: `tick()` advances `ticks` while armed, `fired()`
/// reports whether `after_dynamic` elapsed ticks have accumulated.
///
/// DEVIATION: the logging `name`/`id` fields are dropped (this port logs
/// nothing), `attempts` stays `u8` with wrapping semantics (`+%=`), and
/// `reset_with_jitter` recomputes the jitter range inline instead of via
/// upstream's `range_inclusive(u64, …)`.
#[derive(Clone, Debug)]
pub struct Timeout {
    /// The base timeout in ticks (a multiple of the visible round-trip time).
    after: u64,
    /// The currently armed countdown (`None` iff `!ticking`).
    after_dynamic: Option<u64>,
    /// The number of consecutive times the timeout has fired.
    attempts: u8,
    /// The client's measured round-trip time in ticks.
    rtt: u64,
    /// How many round-trips fit into a single timeout.
    rtt_multiple: u8,
    /// The number of ticks elapsed since (re)arming.
    ticks: u64,
    /// Whether the timeout is currently armed.
    ticking: bool,
}

impl Timeout {
    /// A new timeout armed with `after` ticks and the default round-trip parameters.
    #[must_use]
    pub fn new(after: u64) -> Self {
        Self {
            after,
            after_dynamic: None,
            attempts: 0,
            rtt: constants::RTT_TICKS,
            rtt_multiple: u8::try_from(constants::RTT_MULTIPLE).unwrap_or(u8::MAX),
            ticks: 0,
            ticking: false,
        }
    }

    /// Increments the attempts counter and re-arms with exponential backoff and jitter.
    /// Allows the attempts counter to wrap from time to time.
    /// We do not saturate the counter as this would cause round-robin retries to get stuck.
    ///
    /// # Panics
    /// Panics if the timeout is not currently armed (upstream asserts).
    pub fn backoff(&mut self, prng: &mut Prng) {
        assert!(self.ticking);
        self.ticks = 0;
        self.attempts = self.attempts.wrapping_add(1);
        self.set_after_for_rtt_and_attempts(prng);
    }

    /// It's important to check that when `fired()` is acted on that the timeout is
    /// stopped/started, otherwise further ticks around the event loop may trigger a thundering
    /// herd of messages.
    ///
    /// # Panics
    /// Panics if the timeout fires without being (re)armed (`ticks > after_dynamic`,
    /// upstream "timeout was not reset correctly").
    #[must_use]
    pub fn fired(&self) -> bool {
        match (self.ticking, self.after_dynamic) {
            (true, Some(after)) if self.ticks >= after => {
                assert_eq!(self.ticks, after, "timeout was not reset correctly");
                true
            }
            _ => false,
        }
    }

    /// Stop counting and clear the attempt count (the timeout remains armed).
    ///
    /// # Panics
    /// Panics if the timeout is not currently armed (upstream asserts).
    pub fn reset(&mut self) {
        self.attempts = 0;
        self.ticks = 0;
        assert!(self.ticking);
    }

    /// Re-arm with a countdown uniformly drawn from `[after/2, 3*after/2]`, matching upstream's
    /// `range_inclusive(half, 2*after - half)`.
    ///
    /// # Panics
    /// Panics if the timeout is not armed or `after <= 1` (upstream asserts).
    pub fn reset_with_jitter(&mut self, prng: &mut Prng) {
        self.attempts = self.attempts.wrapping_add(1);
        self.ticks = 0;
        assert!(self.ticking);
        assert!(self.after > 1);
        let half = self.after / 2;
        let range = self.after.saturating_mul(2) - half - half;
        let after_dynamic = half + prng.gen_int_inclusive_u64(range);
        assert!(after_dynamic > 0);
        self.after_dynamic = Some(after_dynamic);
    }

    /// Sets `after` as a function of `rtt` and `attempts`: exponential backoff + jitter.
    /// May be called only after a timeout has been stopped or reset, to prevent backward jumps.
    ///
    /// # Panics
    /// Panics if `ticks != 0` or `rtt == 0` (upstream asserts).
    pub fn set_after_for_rtt_and_attempts(&mut self, prng: &mut Prng) {
        assert_eq!(self.ticks, 0);
        assert!(self.rtt > 0);

        let after = self.rtt.saturating_mul(u64::from(self.rtt_multiple))
            + exponential_backoff_with_jitter(
                prng,
                constants::BACKOFF_MIN_TICKS,
                constants::BACKOFF_MAX_TICKS,
                u64::from(self.attempts),
            );

        assert!(after > 0);
        self.after_dynamic = Some(after);
    }

    /// Recalibrate the round-trip time from a measured ping/pong pair, clamped
    /// into `[1, rtt_max_ticks]` ticks (upstream `vsr.Timeout.set_rtt_ns`).
    ///
    /// # Panics
    /// Panics if the current `rtt` is zero (cannot happen: `rtt` starts at
    /// `constants::rtt_ticks`).
    pub fn set_rtt_ns(&mut self, rtt_ns: u64) {
        assert!(self.rtt > 0);

        // Round to the nearest tick, at least one tick:
        let rtt_ticks = (rtt_ns / 1_000_000 / constants::TICK_MS).max(1);
        let rtt_ticks_clamped = rtt_ticks.min(constants::RTT_MAX_TICKS);
        if self.rtt != rtt_ticks_clamped {
            self.rtt = rtt_ticks_clamped;
        }
    }

    pub fn start(&mut self) {
        self.attempts = 0;
        self.after_dynamic = Some(self.after);
        self.ticks = 0;
        self.ticking = true;
    }

    pub fn stop(&mut self) {
        self.attempts = 0;
        self.after_dynamic = None;
        self.ticks = 0;
        self.ticking = false;
    }

    /// Advance the countdown by one tick (no-op while unarmed).
    pub fn tick(&mut self) {
        if self.ticking {
            self.ticks += 1;
        }
    }
}

/// Calculates exponential backoff with jitter to prevent cascading failure due to thundering
/// herds.
///
/// Upstream: `src/vsr.zig:869` (`exponential_backoff_with_jitter`). The `u128` operand
/// overflows only at `attempt >= 64`, so the exponent saturates at `2^63` full-width lanes.
///
/// # Panics
/// Panics unless `max > min` (upstream asserts).
#[must_use]
pub fn exponential_backoff_with_jitter(prng: &mut Prng, min: u64, max: u64, attempt: u64) -> u64 {
    assert!(max > min);

    // Do not use `@truncate(u6, attempt)` since that only discards the high bits:
    // we want a saturating exponent here instead. (`u6` saturates at 63.)
    let exponent = attempt.min(63);
    // A "1" shifted left gives any power of two; never truncates:
    let power: u128 = 1u128 << exponent;

    // Ensure that `backoff` is calculated correctly when min is 0, taking `@max(1, min)`.
    let min_non_zero = min.max(1);
    assert!(min_non_zero > 0);
    assert!(power > 0);

    // The capped exponential backoff component, `min(range, min_non_zero * 2^attempt)`:
    let backoff_u128 = u128::from(max - min).min(u128::from(min_non_zero).saturating_mul(power));
    // `backoff <= max - min`, so the narrowing cannot lose information.
    #[allow(clippy::cast_possible_truncation)]
    let backoff: u64 = backoff_u128 as u64;
    let jitter = prng.gen_int_inclusive_u64(backoff);

    let result = min + jitter;
    assert!(result >= min);
    assert!(result <= max);

    result
}

/// The client-side of the VSR protocol (upstream `src/vsr/client.zig`).
pub struct Client<B: Bus> {
    /// The outbound bus (upstream: `message_bus`, a full `MessageBus`).
    pub message_bus: B,
    /// The time source (upstream: `vsr.time.Time` vtable).
    pub time: Box<dyn Time>,
    /// A universally unique identifier for the client (must not be zero).
    pub id: u128,
    /// The identifier for the cluster that this client intends to communicate with.
    pub cluster: u128,
    /// The number of replicas in the cluster.
    pub replica_count: u8,
    /// AOF recovery mode: requests carry a fixed `timestamp` so recovery is deterministic.
    pub aof_recovery: bool,
    /// The client's release version.
    pub release: Release,
    /// The total number of ticks elapsed since the client was initialized.
    pub ticks: u64,
    /// The checksum of the latest request/reply, hash-chaining the session.
    pub parent: u128,
    /// The session number (`0` before registration, the register's commit number after).
    pub session: u64,
    /// The request number of the next request.
    pub request_number: u32,
    /// The maximum body size for `command=request` messages, from the register reply.
    pub batch_size_limit: Option<u32>,
    /// The highest view number seen by the client.
    pub view: u32,
    /// The currently processing (non-register) request message.
    pub request_inflight: Option<RequestInflight>,
    /// The retransmission timeout, calibrated from ping/pong round-trip times.
    pub request_timeout: Timeout,
    /// The keepalive / primary-discovery ping cadence.
    pub ping_timeout: Timeout,
    /// The latest estimated round-trip time from each replica.
    pub replica_round_trip_times_ns: [Option<u64>; constants::REPLICAS_MAX],
    /// Used to calculate exponential backoff with random jitter. Seeded with the client's ID.
    pub prng: Prng,
    /// The message pool backing requests.
    pub pool: MessagePool,
    /// Whether this client's session has been evicted.
    pub evicted: bool,
    /// Test hook: called for replies to all operations (including `register`).
    pub on_reply_callback: Option<Box<ReplyCallback<B>>>,
    /// The eviction hook installed at construction (`None` → panics on eviction).
    on_eviction_callback: Option<Box<EvictionCallback<B>>>,
}

impl<B: Bus> Client<B> {
    /// Initialize a new client.
    ///
    /// # Panics
    /// Panics if `id == 0` or `replica_count == 0` (upstream asserts).
    #[must_use]
    pub fn new(
        time: Box<dyn Time>,
        message_bus: B,
        pool: MessagePool,
        options: ClientOptions<B>,
    ) -> Self {
        assert!(options.id > 0);
        assert!(options.replica_count > 0);

        // Upstream: `@truncate(id)` — the low 64 bits seed the PRNG.
        #[allow(clippy::cast_possible_truncation)]
        let prng = Prng::from_seed(options.id as u64);

        let mut ping_timeout = Timeout::new(30_000 / constants::TICK_MS);
        ping_timeout.start();

        Self {
            message_bus,
            time,
            id: options.id,
            cluster: options.cluster,
            replica_count: options.replica_count,
            aof_recovery: options.aof_recovery,
            release: Release::MINIMUM,
            ticks: 0,
            parent: 0,
            session: 0,
            request_number: 0,
            batch_size_limit: None,
            view: 0,
            request_inflight: None,
            request_timeout: Timeout::new(
                constants::RTT_TICKS * u64::try_from(constants::RTT_MULTIPLE).unwrap_or(u64::MAX),
            ),
            ping_timeout,
            replica_round_trip_times_ns: [None; constants::REPLICAS_MAX],
            prng,
            pool,
            evicted: false,
            on_reply_callback: None,
            on_eviction_callback: options.eviction_callback,
        }
    }

    /// Advance the client by one tick (upstream `Client.tick`).
    ///
    /// # Panics
    /// Panics if the client has already been evicted (upstream asserts).
    pub fn tick(&mut self) {
        assert!(!self.evicted);

        self.ticks += 1;

        // Upstream also ticks the message bus (`message_bus.tick_client()`); the
        // sans-IO [`Bus`] has no connection lifecycle to drive.
        self.time.tick();

        self.ping_timeout.tick();
        self.request_timeout.tick();

        if self.ping_timeout.fired() {
            self.on_ping_timeout();
        }
        if self.request_timeout.fired() {
            self.on_request_timeout();
        }
    }

    /// Handle an inbound message from the cluster (upstream `Client.on_message`).
    ///
    /// Ignores messages from a different cluster and misdirected commands; a
    /// client only accepts `.pong_client`, `.reply` and `.eviction`.
    ///
    /// # Panics
    /// Panics if the client has already been evicted (upstream asserts).
    pub fn on_message(&mut self, message: &Message) {
        assert!(!self.evicted);

        let Ok(frame) = <&[u8; message_header::SIZE]>::try_from(message.frame()) else {
            return;
        };
        let Some(frame) = message_header::Header::from_wire(frame) else {
            return;
        };
        if frame.invalid().is_some() {
            return;
        }
        if frame.cluster != self.cluster {
            return;
        }

        match frame.command {
            Command::PongClient => {
                let Some(header) = message.header::<PongClient>() else { return };
                self.on_pong_client(header);
            }
            Command::Reply => {
                let Some(header) = message.header::<Reply>() else { return };
                self.on_reply(message, header);
            }
            Command::Eviction => {
                let Some(header) = message.header::<Eviction>() else { return };
                self.on_eviction(header);
            }
            _ => {
                // Misdirected message; the client ignores it (upstream logs a warning).
            }
        }
    }

    /// Registers a session with the cluster for the client, if this has not yet been done.
    ///
    /// # Panics
    /// Panics if a request is already inflight or the client has already
    /// registered (upstream asserts).
    pub fn register(
        &mut self,
        callback: impl FnMut(u128, &RegisterResult) + 'static,
        user_data: u128,
    ) {
        assert!(!self.evicted);
        assert!(self.request_inflight.is_none());
        assert_eq!(self.request_number, 0);

        let mut message = self.get_message();
        let header = Request {
            client: self.id,
            request: self.request_number,
            cluster: self.cluster,
            operation: Operation::REGISTER,
            release: self.release,
            size: SIZE_U32 + u32::try_from(size_of::<RegisterRequest>()).unwrap_or(u32::MAX),
            // During AOF recovery, if we were to pass timestamp=0, the primary would assign the
            // timestamp. Instead, we send a fixed bogus timestamp (1), to ensure that AOF
            // recovery is deterministic.
            timestamp: u64::from(self.aof_recovery),
            ..Request::default()
        };
        // We will set parent, session, view and checksums only when sending for the first time:
        let body = RegisterRequest { batch_size_limit: 0, reserved: [0; 252] };
        let body_bytes = {
            let mut bytes = [0_u8; size_of::<RegisterRequest>()];
            bytes[..4].copy_from_slice(&body.batch_size_limit.to_le_bytes());
            bytes
        };
        message.set_body(&body_bytes);
        message.set_header(&header);

        assert_eq!(self.request_number, 0);
        self.request_number += 1;

        self.request_inflight = Some(RequestInflight {
            message,
            user_data,
            callback: Callback::Register(Box::new(callback)),
        });
        self.send_request_for_the_first_time();

        // Proactively send ping to replicas so they can identify this peer
        // (see `recv_update_peer` in MessageBus).
        self.on_ping_timeout();
    }

    /// Sends a request message with the operation and events payload to the replica.
    /// There must be no other request message currently inflight.
    ///
    /// # Panics
    /// Panics unless the session is registered, the body fits the batch size
    /// limit, and the body is a whole number of events (upstream asserts).
    pub fn request(
        &mut self,
        callback: impl FnMut(u128, Operation, u64, &[u8]) + 'static,
        user_data: u128,
        operation: Operation,
        events: &[u8],
    ) {
        assert!(!self.evicted);
        assert!(self.request_inflight.is_none());
        assert!(self.request_number > 0);
        assert!(events.len() <= constants::MESSAGE_BODY_SIZE_MAX);
        let batch_size_limit = self
            .batch_size_limit
            .unwrap_or_else(|| panic!("the session must be registered before issuing requests"));
        assert!(events.len() <= usize::try_from(batch_size_limit).unwrap_or(usize::MAX));
        let event_size = event_size(operation);
        assert_eq!(events.len() % event_size, 0);

        let mut message = self.get_message();
        let header = Request {
            client: self.id,
            cluster: self.cluster,
            operation,
            release: self.release,
            size: SIZE_U32 + u32::try_from(events.len()).unwrap_or(u32::MAX),
            ..Request::default()
        };
        // parent/session/view/request are stamped on the first send
        // (`send_request_for_the_first_time`), and the checksums are computed there too.
        message.set_body(events);
        message.set_header(&header);

        self.raw_request(callback, user_data, message);
    }

    /// Sends a request, only setting request_number in the header.
    /// There must be no other request message currently inflight.
    ///
    /// # Panics
    /// Panics unless `request_number > 0`, the message header is a well-formed
    /// request for this client/session, and no request is already inflight
    /// (upstream asserts).
    pub fn raw_request(
        &mut self,
        callback: impl FnMut(u128, Operation, u64, &[u8]) + 'static,
        user_data: u128,
        mut message: Message,
    ) {
        assert!(self.request_inflight.is_none());
        assert!(self.request_number > 0);

        let Some(mut header) = message.header::<Request>() else {
            unreachable!("the message is a request");
        };
        assert_eq!(header.client, self.id);
        assert_eq!(header.release.value, self.release.value);
        assert_eq!(header.cluster, self.cluster);
        assert_eq!(header.command, Command::Request);
        assert!(header.size >= SIZE_U32);
        assert!(header.size <= constants::MESSAGE_SIZE_MAX);
        let batch_size_limit = self
            .batch_size_limit
            .unwrap_or_else(|| panic!("the session must be registered before issuing requests"));
        assert!(header.size <= SIZE_U32 + batch_size_limit);
        // Upstream: `assert(message.header.operation.valid(Operation))` — the state-machine
        // operation set is not yet available in the vsr crate, so we approximate with the
        // vsr-level checks (DEVIATION).
        assert_eq!(header.view, 0);
        assert_eq!(header.parent, 0);
        assert_eq!(header.session, 0);
        assert_eq!(header.request, 0);
        assert_ne!((header.timestamp == 0), self.aof_recovery);
        if !self.aof_recovery {
            assert!(header.operation == Operation::NOOP || !header.operation.vsr_reserved());
        }

        header.request = self.request_number;
        self.request_number += 1;
        message.set_header(&header);

        self.request_inflight = Some(RequestInflight {
            message,
            user_data,
            callback: Callback::Request(Box::new(callback)),
        });
        self.send_request_for_the_first_time();
    }

    /// Acquires a message from the message pool.
    /// Either use it in [`Self::raw_request`] or discard via [`Self::release_message`].
    ///
    /// # Panics
    /// Panics if the pool is exhausted (upstream asserts).
    #[must_use]
    pub fn get_message(&mut self) -> Message {
        self.pool.get_message()
    }

    /// Releases a message back to the message pool.
    ///
    /// # Panics
    /// Panics if no slot was awaiting a release (i.e., more releases than checkouts).
    pub fn release_message(&mut self, message: Message) {
        self.pool.release(message);
    }

    fn on_eviction(&mut self, eviction: Eviction) {
        assert!(!self.evicted);
        assert_eq!(eviction.command, Command::Eviction);
        assert_eq!(eviction.cluster, self.cluster);

        if eviction.client != self.id {
            return;
        }
        if eviction.view < self.view {
            return;
        }

        assert_eq!(eviction.client, self.id);
        assert!(eviction.view >= self.view);

        let callback = self.on_eviction_callback.take();
        if let Some(mut callback) = callback {
            self.evicted = true;
            callback(self, &eviction);
        } else {
            panic!(
                "session evicted: reason={:?} (cluster_release={:?})",
                eviction.reason(),
                eviction.release
            );
        }
    }

    #[allow(clippy::similar_names)] // upstream: ping_timestamp_monotonic / pong_timestamp_monotonic
    fn on_pong_client(&mut self, pong: PongClient) {
        assert_eq!(pong.command, Command::PongClient);
        assert_eq!(pong.cluster, self.cluster);

        if pong.view > self.view {
            // Even if there is a request in flight, don't try to retransmit it immediately
            // after a view change. Instead, ride the on_request_timeout normally to reduce the
            // size of thundering herd.
            // Upstream `maybe(self.request_inflight != null)` is a no-op assertion.
            self.view = pong.view;
        }

        let ping_timestamp_monotonic = pong.ping_timestamp_monotonic;
        let pong_timestamp_monotonic = self.time.monotonic().ns;
        if ping_timestamp_monotonic <= pong_timestamp_monotonic {
            self.replica_round_trip_times_ns[pong.replica as usize] =
                Some(pong_timestamp_monotonic - ping_timestamp_monotonic);

            let mut round_trip_times_ns: Vec<u64> =
                self.replica_round_trip_times_ns.iter().filter_map(|rtt_ns| *rtt_ns).collect();
            round_trip_times_ns.sort_unstable();
            assert!(!round_trip_times_ns.is_empty());

            let rtt_median_ns = round_trip_times_ns[round_trip_times_ns.len() / 2];
            self.request_timeout.set_rtt_ns(rtt_median_ns);
        }
    }

    fn on_reply(&mut self, reply_message: &Message, header: Reply) {
        // We check these checksums again here because this is the last time we get to downgrade
        // a correctness bug into a liveness bug, before we return data back to the application.
        assert!(header.valid_checksum());
        assert!(header.valid_checksum_body(reply_message.body_used()));
        assert_eq!(header.command, Command::Reply);
        assert_eq!(header.release.value, self.release.value);

        if header.client != self.id {
            return;
        }

        let Some(inflight) = self.request_inflight.as_ref() else {
            assert!(header.request < self.request_number);
            return;
        };
        // Upstream keeps the inflight header read before consuming the request.
        let Some(inflight_header) = inflight.message.header::<Request>() else {
            unreachable!("the inflight message is a request");
        };
        if header.request < inflight_header.request {
            assert!(inflight_header.request > 0);
            assert_ne!(inflight_header.operation, Operation::REGISTER);
            return;
        }

        assert_eq!(header.request, inflight_header.request);
        assert_eq!(header.request_checksum, inflight_header.checksum());
        let inflight_vsr_operation = inflight_header.operation;
        let inflight_request = inflight_header.request;
        if inflight_vsr_operation == Operation::REGISTER {
            assert_eq!(inflight_request, 0);
        } else {
            assert!(inflight_request > 0);
        }

        // Consume the inflight request here before invoking callbacks down below in case they
        // wish to queue a new `request_inflight`.
        let Some(inflight) = self.request_inflight.take() else {
            unreachable!("inflight request was just present");
        };
        let callback = inflight.callback;
        let user_data = inflight.user_data;

        if let Some(mut on_reply_callback) = self.on_reply_callback.take() {
            on_reply_callback(self, &inflight.message, reply_message);
            self.on_reply_callback = Some(on_reply_callback);
        }

        assert_eq!(header.request_checksum, self.parent);
        assert_eq!(header.client, self.id);
        assert_eq!(header.request, inflight_request);
        assert_eq!(header.cluster, self.cluster);
        assert_eq!(header.op, header.commit);
        assert_eq!(header.operation, inflight_vsr_operation);

        // The context of this reply becomes the parent of our next request:
        self.parent = header.context;

        if header.view > self.view {
            self.view = header.view;
        }

        self.request_timeout.stop();

        // Release request message to ensure that inflight's callback can submit a new one.
        self.release_message(inflight.message);

        if inflight_vsr_operation == Operation::REGISTER {
            assert_eq!(inflight_request, 0);
            assert!(self.batch_size_limit.is_none());
            assert_eq!(self.session, 0);
            assert!(header.commit > 0);
            assert_eq!(header.size, SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32);

            let mut result_bytes = [0_u8; 4];
            result_bytes.copy_from_slice(&reply_message.body_used()[..4]);
            let result = RegisterResult {
                batch_size_limit: u32::from_le_bytes(result_bytes),
                reserved: [0; 252],
            };
            assert!(result.batch_size_limit > 0);
            assert!(
                result.batch_size_limit
                    <= u32::try_from(constants::MESSAGE_BODY_SIZE_MAX).unwrap_or(u32::MAX)
            );

            self.session = header.commit; // The commit number becomes the session number.
            self.batch_size_limit = Some(result.batch_size_limit);
            match callback {
                Callback::Register(mut callback) => callback(user_data, &result),
                Callback::Request(_) => {
                    unreachable!("a register inflight carries a register callback")
                }
            }
        } else {
            // The message is the result of raw_request(), so invoke the user callback.
            // NOTE: the callback is allowed to mutate `reply.body_used()` here.
            match callback {
                Callback::Request(mut callback) => callback(
                    user_data,
                    inflight_vsr_operation,
                    header.timestamp,
                    reply_message.body_used(),
                ),
                Callback::Register(_) => {
                    unreachable!("a request inflight carries a request callback")
                }
            }
        }
    }

    fn on_ping_timeout(&mut self) {
        self.ping_timeout.reset();

        let ping = PingClient {
            cluster: self.cluster,
            release: self.release,
            client: self.id,
            ping_timestamp_monotonic: self.time.monotonic().ns,
            session: self.session,
            ..PingClient::default()
        };

        self.send_header_to_replicas(&ping);
    }

    // Possible reasons for a timeout:
    // - the cluster is overloaded and takes too long to respond
    // - the request message got dropped by the network
    // - there was a view change, and we are not speaking to the primary
    fn on_request_timeout(&mut self) {
        self.request_timeout.backoff(&mut self.prng); // Reduce the load.

        let message = match self.request_inflight.as_ref() {
            Some(inflight) => inflight.message.clone(),
            None => unreachable!("an inflight request"),
        };
        let Some(header) = message.header::<Request>() else {
            unreachable!("a request");
        };
        assert_eq!(header.command, Command::Request);
        assert!(header.request < self.request_number);
        assert_eq!(header.checksum(), self.parent);
        assert_eq!(header.session, self.session);

        self.send_request_with_hedging();
    }

    fn send_header_to_replicas(&mut self, header: &PingClient) {
        // Upstream builds the message from the header, computing the (empty) body and header
        // checksums (`create_message_from_header`), so the sent frame always checksums.
        let mut header = *header;
        header.set_checksum_body(&[]);
        header.set_checksum();

        let mut message = self.get_message();
        message.set_body(&[]);
        message.set_header(&header);
        self.send_message_to_replicas(&message);
        self.release_message(message);
    }

    fn send_message_to_replicas(&mut self, message: &Message) {
        for replica in 0..self.replica_count {
            self.send_message_to_replica(replica, message);
        }
    }

    fn send_message_to_replica(&mut self, replica: u8, message: &Message) {
        assert!(replica < self.replica_count);

        // The client only writes `.request` and `.ping_client` frames, both addressed to
        // itself; upstream asserts the same inside `send_message_to_replica`.
        if let Some(request) = message.header::<Request>() {
            assert!(request.valid_checksum());
            assert_eq!(request.cluster, self.cluster);
            assert_eq!(request.client, self.id);
        } else {
            let Some(ping) = message.header::<PingClient>() else {
                unreachable!("client sends only request/ping_client messages");
            };
            assert!(ping.valid_checksum());
            assert_eq!(ping.cluster, self.cluster);
            assert_eq!(ping.client, self.id);
        }

        self.message_bus.send(replica, message);
    }

    // In addition to the primary, each request is also sent to a randomly chosen backup, to
    // handle the case where the client → primary link is down. This ensures logical
    // availability of the cluster, i.e., as long the client is connected to a backup that in
    // turn is connected to the primary, the request will be processed by the cluster.
    fn send_request_with_hedging(&mut self) {
        let message = match self.request_inflight.as_ref() {
            Some(inflight) => inflight.message.clone(),
            None => unreachable!("an inflight request"),
        };
        let primary = u8::try_from(self.view % u32::from(self.replica_count))
            .unwrap_or_else(|_| unreachable!("primary < replica_count {}", self.replica_count));
        self.send_message_to_replica(primary, &message);

        if self.replica_count > 1 {
            // The offset is drawn from [1, replica_count - 1] so the backup is never the primary:
            let offset = 1 + self.prng.gen_int_inclusive_u8(self.replica_count - 2);
            let backup = (primary + offset) % self.replica_count;
            assert_ne!(backup, primary);
            self.send_message_to_replica(backup, &message);
        }
    }

    // We set the message checksums only when sending the request for the first time,
    // which is when we have the checksum of the latest reply available to set as `parent`,
    // and similarly also the session number if requests were queued while registering:
    fn send_request_for_the_first_time(&mut self) {
        assert!(self.request_number > 0);

        let checksum = {
            let Some(message) =
                self.request_inflight.as_mut().map(|inflight| &mut inflight.message)
            else {
                unreachable!("an inflight request");
            };
            let Some(mut header) = message.header::<Request>() else {
                unreachable!("a request");
            };
            assert_eq!(header.command, Command::Request);
            assert_eq!(header.parent, 0);
            assert_eq!(header.session, 0);
            assert!(header.request < self.request_number);
            assert_eq!(header.view, 0);
            assert!(header.size <= constants::MESSAGE_SIZE_MAX);

            header.parent = self.parent;
            header.session = self.session;
            // We also try to include our highest view number, so we wait until the request is
            // ready to be sent for the first time. However, beyond that, it is not necessary to
            // update the view number again, for example if it should change between now and
            // resending.
            header.view = self.view;
            let body = message.body_used().to_vec();
            header.set_checksum_body(&body);
            header.set_checksum();
            message.set_header(&header);
            header.checksum()
        };

        // The checksum of this request becomes the parent of our next reply:
        self.parent = checksum;

        assert!(!self.request_timeout.ticking);
        self.request_timeout.start();

        self.send_request_with_hedging();
    }
}

/// The size in bytes of a single event for `operation` (upstream
/// `StateMachineOperation.event_size`, injected via the comptime type parameter
/// of `ClientType`).
///
/// DEVIATION: upstream derives this from the client's generic
/// `StateMachineOperation`; here the vsr-level [`Operation`] carries the sizes.
///
/// # Panics
/// Panics if `operation` is not one of the client-facing state-machine operations.
#[must_use]
pub fn event_size(operation: Operation) -> usize {
    match operation {
        Operation::CREATE_ACCOUNTS => size_of::<Account>(),
        Operation::CREATE_TRANSFERS => size_of::<Transfer>(),
        Operation::LOOKUP_ACCOUNTS | Operation::LOOKUP_TRANSFERS => 16,
        Operation::GET_ACCOUNT_TRANSFERS | Operation::GET_ACCOUNT_BALANCES => {
            size_of::<AccountFilter>()
        }
        Operation::GET_CHANGE_EVENTS => size_of::<ChangeEventsFilter>(),
        Operation::QUERY_ACCOUNTS | Operation::QUERY_TRANSFERS => size_of::<QueryFilter>(),
        operation => panic!("operation {} is not a client request operation", operation.0),
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::message_header::Reason;
    use crate::testing::time::{OffsetType, TimeSim};

    use super::*;

    const CLIENT_ID: u128 = 0x0102_0304_0506_0708;
    const TRANSFER_SIZE: usize = size_of::<tigerbeetle_core::types::Transfer>();
    const CLUSTER: u128 = 0x1234_5678;
    const REPLICA_COUNT: u8 = 3;

    /// A bus that records every outbound message (upstream's test `MessageBus`).
    #[derive(Default)]
    struct RecorderBus {
        sent: Vec<(u8, Message)>,
    }

    impl Bus for RecorderBus {
        fn send(&mut self, replica: u8, message: &Message) {
            self.sent.push((replica, message.clone()));
        }
    }

    fn new_client() -> Client<RecorderBus> {
        let time = Box::new(TimeSim::new(1, OffsetType::Linear, 0, 0));
        let pool = MessagePool::init_capacity(16);
        Client::new(
            time,
            RecorderBus::default(),
            pool,
            ClientOptions {
                id: CLIENT_ID,
                cluster: CLUSTER,
                replica_count: REPLICA_COUNT,
                aof_recovery: false,
                eviction_callback: None,
            },
        )
    }

    /// Builds a reply frame for the given request message, echoing `result` bodies for
    /// registers (`op == commit == session`) and `timestamp` for requests.
    fn build_reply(
        request: &Request,
        op: u64,
        context: u128,
        timestamp: u64,
        body: &[u8],
    ) -> Message {
        let mut message = Message::new();
        let mut header = Reply {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            client: CLIENT_ID,
            request_checksum: request.checksum(),
            context,
            op,
            commit: op,
            timestamp,
            request: request.request,
            operation: request.operation,
            size: SIZE_U32 + u32::try_from(body.len()).unwrap_or(u32::MAX),
            ..Reply::default()
        };
        header.set_checksum_body(body);
        header.set_checksum();
        message.set_body(body);
        message.set_header(&header);
        message
    }

    /// The register reply body: a single `RegisterResult`.
    fn register_result_body(batch_size_limit: u32) -> Vec<u8> {
        let result = RegisterResult { batch_size_limit, reserved: [0; 252] };
        let mut bytes = vec![0_u8; size_of::<RegisterResult>()];
        bytes[..4].copy_from_slice(&result.batch_size_limit.to_le_bytes());
        bytes
    }

    fn sent_request(bus: &RecorderBus) -> (&u8, Request) {
        bus.sent
            .iter()
            .filter_map(|(replica, message)| {
                message.header::<Request>().map(|request| (replica, request))
            })
            .next_back()
            .unwrap()
    }

    /// The primary's copy of the first (current) request.
    fn primary_request(bus: &RecorderBus) -> (&u8, Request) {
        bus.sent
            .iter()
            .find_map(|(replica, message)| {
                message.header::<Request>().map(|request| (replica, request))
            })
            .unwrap()
    }

    fn count_requests(bus: &RecorderBus) -> usize {
        bus.sent.iter().filter(|(_, message)| message.header::<Request>().is_some()).count()
    }

    #[test]
    fn register_then_request_round_trip() {
        let mut client = new_client();

        let register_result = Rc::new(RefCell::new(None));
        let register_handle = Rc::clone(&register_result);
        client.register(
            move |user_data, result: &RegisterResult| {
                *register_handle.borrow_mut() = Some((user_data, result.batch_size_limit));
            },
            42,
        );

        // `register` sends the request and pings each replica:
        assert_eq!(primary_request(&client.message_bus).0, &0);
        assert_eq!(count_requests(&client.message_bus), 2);
        let pings = client
            .message_bus
            .sent
            .iter()
            .filter(|(_, message)| message.header::<PingClient>().is_some())
            .count();
        assert_eq!(pings, 3);
        assert!(client.request_inflight.is_some());

        // The first request is the register: session=0, parent=0, operation=register.
        let (_, register) = sent_request(&client.message_bus);
        assert_eq!(register.client, CLIENT_ID);
        assert_eq!(register.session, 0);
        assert_eq!(register.parent, 0);
        assert_eq!(register.operation, Operation::REGISTER);
        assert!(register.valid_checksum());
        assert_eq!(
            register.size,
            SIZE_U32 + u32::try_from(size_of::<RegisterResult>()).unwrap_or(u32::MAX)
        );

        // Deliver the register reply (commit=1 becomes the session number).
        let reply = build_reply(&register, 1, 0xaaaa, 2, &register_result_body(1024));
        client.on_message(&reply);
        assert_eq!(client.session, 1);
        assert_eq!(client.batch_size_limit, Some(1024));
        assert_eq!(*register_result.borrow(), Some((42, 1024)));
        assert!(client.request_inflight.is_none());
        assert_eq!(client.parent, 0xaaaa);

        // Issue a (single-transfer) request on the registered session.
        let request_result = Rc::new(RefCell::new(None));
        let request_handle = Rc::clone(&request_result);
        let events = vec![0_u8; TRANSFER_SIZE];
        client.request(
            move |user_data, operation, timestamp, body| {
                *request_handle.borrow_mut() =
                    Some((user_data, operation, timestamp, body.to_vec()));
            },
            7,
            Operation::CREATE_TRANSFERS,
            &events,
        );

        let requests: Vec<(u8, Request)> = client
            .message_bus
            .sent
            .iter()
            .filter_map(|(replica, message)| {
                message.header::<Request>().map(|request| (*replica, request))
            })
            .collect();
        // The primary's copy of the transfer (highest `request` number send to replica 0):
        let (primary, request) = requests
            .iter()
            .filter(|(replica, _)| *replica == 0)
            .max_by_key(|(_, request)| request.request)
            .copied()
            .unwrap();
        assert_eq!(primary, 0);
        assert_eq!(request.client, CLIENT_ID);
        assert_eq!(request.session, 1);
        assert_eq!(request.operation, Operation::CREATE_TRANSFERS);
        assert!(request.valid_checksum());
        assert_eq!(request.parent, 0xaaaa);

        // Deliver the request reply; `context` chains the next request.
        let body = vec![1_u8, 2, 3, 4];
        let reply = build_reply(&request, 2, 0xbbbb, 9876, &body);
        client.on_message(&reply);
        assert_eq!(*request_result.borrow(), Some((7, Operation::CREATE_TRANSFERS, 9876, body)));
        assert_eq!(client.parent, 0xbbbb);
        assert_eq!(client.request_number, 2);
        assert!(client.request_inflight.is_none());
    }

    #[test]
    fn hedging_sends_to_primary_and_a_distinct_backup() {
        let mut client = new_client();
        let register_result = Rc::new(RefCell::new(None));
        let handle = Rc::clone(&register_result);
        client.register(
            move |user_data, result: &RegisterResult| {
                *handle.borrow_mut() = Some((user_data, result.batch_size_limit));
            },
            0,
        );

        let requests: Vec<(u8, Request)> = client
            .message_bus
            .sent
            .iter()
            .filter_map(|(replica, message)| {
                message.header::<Request>().map(|request| (*replica, request))
            })
            .collect();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].0, 0, "primary is replica 0 (view=0)");
        assert_ne!(requests[1].0, requests[0].0, "hedge must beat the primary");
        assert!(requests[1].0 < REPLICA_COUNT);

        // The two hedged copies are byte-identical (same checksum).
        assert_eq!(requests[0].1.checksum(), requests[1].1.checksum());

        // Release the inflight request before the pool asserts all messages are returned.
        let (_, register) = sent_request(&client.message_bus);
        let reply = build_reply(&register, 1, 0xdddd, 2, &register_result_body(1024));
        client.on_message(&reply);
        assert!(client.request_inflight.is_none());
    }

    #[test]
    fn request_timeout_retransmits_with_backoff() {
        let mut client = new_client();
        client.register(|_user_data, _result: &RegisterResult| {}, 0);

        // Deliver the register reply so we are past registration.
        let (_, register) = sent_request(&client.message_bus);
        let reply = build_reply(&register, 1, 0xaaaa, 2, &register_result_body(1024));
        client.on_message(&reply);

        // Issue a request but never answer it — the request timeout must retransmit.
        let events = vec![0_u8; TRANSFER_SIZE];
        client.request(
            |_user_data, _operation, _timestamp, _body| {},
            5,
            Operation::CREATE_TRANSFERS,
            &events,
        );

        let sent_before = count_requests(&client.message_bus);
        let request_number_before = client.request_number;

        // Register's ping fires at 30s; the request timeout fires at rtt*2 = 60 ticks.
        for _ in 0..=61 {
            client.tick();
        }

        let sent_after = count_requests(&client.message_bus);
        assert!(
            sent_after > sent_before,
            "timeout must retransmit the request ({sent_before} -> {sent_after})"
        );
        assert_eq!(
            client.request_number, request_number_before,
            "resend reuses the same request number"
        );

        // Each retransmission hits the primary plus one backup.
        let requests: Vec<&u8> = client
            .message_bus
            .sent
            .iter()
            .filter(|(_, message)| message.header::<Request>().is_some())
            .map(|(replica, _)| replica)
            .collect();
        let resent: Vec<u8> = requests.into_iter().skip(sent_before).copied().collect();
        assert_eq!(resent.len(), 2);
        assert_ne!(resent[1], resent[0]);

        // Answer the retransmission to release the inflight request back to the pool.
        let (_, request) = sent_request(&client.message_bus);
        let reply = build_reply(&request, 2, 0xbbbb, 9876, &[0_u8; 4]);
        client.on_message(&reply);
        assert!(client.request_inflight.is_none());
    }

    #[test]
    fn pong_calibrates_rtt() {
        let mut client = new_client();

        // Same-tick ping we "sent" now: rtt = 0.
        let ping_timestamp_monotonic = client.time.monotonic().ns;
        let mut pong = PongClient {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            ping_timestamp_monotonic,
            replica: 2,
            ..PongClient::default()
        };
        pong.set_checksum_body(&[]);
        pong.set_checksum();
        let mut message = Message::new();
        message.set_body(&[]);
        message.set_header(&pong);

        client.on_message(&message);
        assert_eq!(client.replica_round_trip_times_ns[2], Some(0));

        // A second sample yields a median (single-element filter here).
        client.on_message(&message);
        assert_eq!(client.replica_round_trip_times_ns[2], Some(0));
    }

    #[test]
    fn newer_pong_view_bumps_client_view() {
        let mut client = new_client();
        client.view = 0;

        let mut pong = PongClient {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            ping_timestamp_monotonic: client.time.monotonic().ns,
            view: 7,
            ..PongClient::default()
        };
        pong.set_checksum_body(&[]);
        pong.set_checksum();
        let mut message = Message::new();
        message.set_body(&[]);
        message.set_header(&pong);

        client.on_message(&message);
        assert_eq!(client.view, 7);
    }

    #[test]
    fn wrong_cluster_messages_are_ignored() {
        let mut client = new_client();
        let mut pong = PongClient {
            cluster: CLUSTER + 1,
            release: Release::MINIMUM,
            ping_timestamp_monotonic: client.time.monotonic().ns,
            ..PongClient::default()
        };
        pong.set_checksum_body(&[]);
        pong.set_checksum();
        let mut message = Message::new();
        message.set_body(&[]);
        message.set_header(&pong);

        client.on_message(&message);
        assert_eq!(client.replica_round_trip_times_ns[0], None);
    }

    #[test]
    fn eviction_invokes_callback() {
        let evicted = Rc::new(RefCell::new(None));
        let handle = Rc::clone(&evicted);
        let mut client = Client::new(
            Box::new(TimeSim::new(1, OffsetType::Linear, 0, 0)),
            RecorderBus::default(),
            MessagePool::init_capacity(16),
            ClientOptions {
                id: CLIENT_ID,
                cluster: CLUSTER,
                replica_count: REPLICA_COUNT,
                aof_recovery: false,
                eviction_callback: Some(Box::new(
                    move |_client: &mut Client<RecorderBus>, eviction: &Eviction| {
                        *handle.borrow_mut() = eviction.reason();
                    },
                )),
            },
        );

        let mut eviction = Eviction {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            client: CLIENT_ID,
            reason_ordinal: 1, // NoSession
            ..Eviction::default()
        };
        eviction.set_checksum_body(&[]);
        eviction.set_checksum();
        let mut message = Message::new();
        message.set_body(&[]);
        message.set_header(&eviction);

        client.on_message(&message);
        assert!(client.evicted);
        assert_eq!(*evicted.borrow(), Some(Reason::NoSession));
    }

    #[test]
    fn eviction_without_callback_panics() {
        let mut client = new_client();

        let mut eviction = Eviction {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            client: CLIENT_ID,
            reason_ordinal: 4, // InvalidRequestOperation
            ..Eviction::default()
        };
        eviction.set_checksum_body(&[]);
        eviction.set_checksum();
        let mut message = Message::new();
        message.set_body(&[]);
        message.set_header(&eviction);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.on_message(&message);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn event_size_matches_schema() {
        use tigerbeetle_core::types::{
            Account, AccountFilter, ChangeEventsFilter, QueryFilter, Transfer,
        };
        assert_eq!(event_size(Operation::CREATE_ACCOUNTS), size_of::<Account>());
        assert_eq!(event_size(Operation::CREATE_TRANSFERS), size_of::<Transfer>());
        assert_eq!(event_size(Operation::LOOKUP_ACCOUNTS), 16);
        assert_eq!(event_size(Operation::GET_ACCOUNT_TRANSFERS), size_of::<AccountFilter>());
        assert_eq!(event_size(Operation::GET_CHANGE_EVENTS), size_of::<ChangeEventsFilter>());
        assert_eq!(event_size(Operation::QUERY_TRANSFERS), size_of::<QueryFilter>());
    }

    #[test]
    fn timeout_and_backoff_semantics() {
        let mut prng = Prng::from_seed(0);
        let mut timeout = Timeout::new(10);
        assert!(!timeout.fired());

        timeout.start();
        for _ in 0..10 {
            timeout.tick();
        }
        assert!(timeout.fired());

        timeout.reset();
        assert!(!timeout.fired());
        timeout.backoff(&mut prng);
        // The backoff lengthens the timeout beyond the base rtt*2 = 60 ticks:
        for _ in 0..=10 {
            timeout.tick();
        }
        assert!(!timeout.fired(), "backoff lengthens the timeout");

        // reset_with_jitter stays within [half, 3/2*after]:
        let mut timeout = Timeout::new(100);
        timeout.start();
        timeout.reset_with_jitter(&mut prng);
        let after = timeout.after_dynamic.unwrap();
        assert!((50..=150).contains(&after));
    }

    #[test]
    fn exponential_backoff_bounds_with_jitter() {
        let mut prng = Prng::from_seed(1);
        for attempt in 0..70 {
            let backoff = exponential_backoff_with_jitter(&mut prng, 1, 1000, attempt);
            assert!((1..=1000).contains(&backoff), "attempt={attempt} backoff={backoff}");
        }
    }
}
