//! Utils functions for writing fuzzers (upstream: `src/testing/fuzz.zig`).
//!
//! DEVIATION: test-only code may use floats (upstream does too here); the no-float rule
//! applies to state transitions, not distribution sampling in fuzzers.

#![allow(clippy::cast_possible_truncation)] // lossyCast semantics are intentional
#![allow(clippy::cast_precision_loss)] // widening into f64 for distribution math
#![allow(clippy::cast_sign_loss)] // exponential samples are non-negative by construction
#![allow(clippy::cast_lossless)] // macro covers u32/u64/usize; only u32 has From<f64>

use crate::stdx::prng::Prng;

/// Integer types usable with [`random_int_exponential`] (upstream is comptime-generic).
pub trait FuzzInt: Copy {
    const MAX: Self;

    fn from_u64(value: u64) -> Self;

    /// Lossy widening cast used only for distribution math (test-only).
    fn approx_to_f64(self) -> f64;
}

macro_rules! impl_fuzz_int {
    ($($t:ty),*) => {$(
        impl FuzzInt for $t {
            const MAX: Self = <$t>::MAX;

            fn from_u64(value: u64) -> Self {
                value as $t
            }

            fn approx_to_f64(self) -> f64 {
                // u64/usize have no lossless From<f64> path; widening cast is fine here.
                self as f64
            }
        }
    )*};
}

impl_fuzz_int!(u32, u64, usize);

/// Returns an integer of type `T` with an exponential distribution of rate `avg`.
/// Note: If you specify a very high rate then `T::MAX` may be over-represented.
///
/// DEVIATION: upstream draws `floatExp(f64)` from the PRNG byte stream; we derive the
/// exponential sample with the inverse-CDF method from 53 random bits. Test-only code —
/// distributions match, exact sequences intentionally do not.
#[must_use]
pub fn random_int_exponential<T: FuzzInt>(prng: &mut Prng, avg: T) -> T {
    let bits = prng.int_u64();
    // 53-bit uniform in [0, 1):
    let uniform = (bits >> 11) as f64 / (1_u64 << 53) as f64;
    // Guard against ln(0); the probability of a zero draw is negligible.
    let uniform = if uniform == 0.0 { f64::EPSILON } else { uniform };
    let exp = -uniform.ln() * avg.approx_to_f64();
    if !exp.is_finite() || exp >= T::MAX.approx_to_f64() { T::MAX } else { T::from_u64(exp as u64) }
}
