//! Port of `src/vsr/checksum.zig` (and the Aegis-128L MAC it is built on, vendored upstream at
//! `src/stdx/vendored/aegis.zig` from Zig 0.13.0's standard library).
//!
//! This checksum:
//! - detects bitrot in data on disk,
//! - validates network messages before casting raw bytes to a wire-format struct,
//! - hash-chains prepares and client requests to have strong consistency and ordering guarantees.
//!
//! As this checksum is stored on disk, it is set in stone and impossible to change.
//!
//! Upstream uses the hardware-accelerated AES block instruction (`vaesenc`) via Zig std.
//!
//! DEVIATION: this port implements the AES round function in software (`aes_round`), because
//! `unsafe` is forbidden by repo policy and stable Rust exposes no safe AES-NI wrapper. The
//! output is bit-for-bit identical to upstream — verified by the ported test vectors, the
//! "checksum stability" change-detector hash, and the alignment/sizing hash below.
//! TODO(port): performance — consider an isolated unsafe-free SIMD abstraction once available;
//! until then this implementation prioritizes correctness over speed.

#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream

// ---------------------------------------------------------------------------
// AES single-round primitive (software).
//
// AESRound(state, round_key) = MixColumns(ShiftRows(SubBytes(state))) ^ round_key,
// identical to the x86 `vaesenc` semantics used by AEGIS.
// ---------------------------------------------------------------------------

/// The AES S-box, computed at compile time from GF(2^8) inversion + the affine transform
/// (avoids transcribing 256 constants; verified against known values below).
const SBOX: [u8; 256] = build_sbox();

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // x^8 + x^4 + x^3 + x + 1
        }
        b >>= 1;
    }
    p
}

const fn rotl8(b: u8, n: u32) -> u8 {
    b.rotate_left(n)
}

#[allow(clippy::cast_possible_truncation)] // loop counters are < 256 by construction
const fn build_sbox() -> [u8; 256] {
    // Discrete log base 3 over GF(2^8)/0x11b; 3 generates the multiplicative group.
    let mut log = [0xFFu8; 256];
    let mut antilog = [0u8; 255];
    let mut e = 1u8;
    let mut j = 0usize;
    while j < 255 {
        log[e as usize] = j as u8;
        antilog[j] = e;
        e = gf_mul(e, 3);
        j += 1;
    }

    let mut sbox = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        // Multiplicative inverse: inv(a) = 3^(255 - log(a)); inv(0) = 0.
        let inverse = if i == 0 { 0u8 } else { antilog[(255 - log[i] as usize) % 255] };
        // Affine transform over GF(2).
        sbox[i] = inverse
            ^ rotl8(inverse, 1)
            ^ rotl8(inverse, 2)
            ^ rotl8(inverse, 3)
            ^ rotl8(inverse, 4)
            ^ 0x63;
        i += 1;
    }
    sbox
}

const _: () = assert!(SBOX[0x00] == 0x63);
const _: () = assert!(SBOX[0x01] == 0x7c);
const _: () = assert!(SBOX[0x53] == 0xed);
const _: () = assert!(SBOX[0xff] == 0x16);

/// One AES encryption round: SubBytes, ShiftRows, MixColumns, XOR round key.
fn aes_round(state: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
    let mut t = [0u8; 16];

    // SubBytes + ShiftRows. Bytes are column-major: state[col * 4 + row].
    for row in 0..4 {
        for col in 0..4 {
            t[col * 4 + row] = SBOX[state[((col + row) & 3) * 4 + row] as usize];
        }
    }

    // MixColumns.
    let mut out = [0u8; 16];
    for col in 0..4 {
        let s0 = t[col * 4];
        let s1 = t[col * 4 + 1];
        let s2 = t[col * 4 + 2];
        let s3 = t[col * 4 + 3];
        out[col * 4] = gf_mul(s0, 2) ^ gf_mul(s1, 3) ^ s2 ^ s3;
        out[col * 4 + 1] = s0 ^ gf_mul(s1, 2) ^ gf_mul(s2, 3) ^ s3;
        out[col * 4 + 2] = s0 ^ s1 ^ gf_mul(s2, 2) ^ gf_mul(s3, 3);
        out[col * 4 + 3] = gf_mul(s0, 3) ^ s1 ^ s2 ^ gf_mul(s3, 2);
    }

    // AddRoundKey.
    for i in 0..16 {
        out[i] ^= round_key[i];
    }
    out
}

const fn xor_blocks(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = a[i] ^ b[i];
        i += 1;
    }
    out
}

fn block_from(src: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&src[..16]);
    out
}

// ---------------------------------------------------------------------------
// AEGIS-128L state. Port of State128L from src/stdx/vendored/aegis.zig.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct State128L {
    blocks: [[u8; 16]; 8],
}

impl State128L {
    fn new(key: &[u8; 16], nonce: &[u8; 16]) -> Self {
        const C1: [u8; 16] = [
            0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, //
            0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd,
        ];
        const C2: [u8; 16] = [
            0x0, 0x1, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, //
            0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62,
        ];

        let key_block = *key;
        let nonce_block = *nonce;
        let key_xor_nonce = xor_blocks(&key_block, &nonce_block);

        let mut state = Self {
            blocks: [
                key_xor_nonce,
                C1,
                C2,
                C1,
                key_xor_nonce,
                xor_blocks(&key_block, &C2),
                xor_blocks(&key_block, &C1),
                xor_blocks(&key_block, &C2),
            ],
        };

        let mut i = 0;
        while i < 10 {
            state.update(&nonce_block, &key_block);
            i += 1;
        }
        state
    }

    fn update(&mut self, d1: &[u8; 16], d2: &[u8; 16]) {
        // Hoist lanes; this keeps the blocks on the stack/registers like upstream does.
        let mut blocks = self.blocks;
        let tmp = blocks[7];

        for i in (1..8).rev() {
            blocks[i] = aes_round(&blocks[i - 1], &blocks[i]);
        }

        blocks[0] = aes_round(&tmp, &blocks[0]);
        blocks[0] = xor_blocks(&blocks[0], d1);
        blocks[4] = xor_blocks(&blocks[4], d2);

        self.blocks = blocks;
    }

    fn absorb(&mut self, src: &[u8; 32]) {
        let msg0 = block_from(&src[0..16]);
        let msg1 = block_from(&src[16..32]);
        self.update(&msg0, &msg1);
    }

    /// Authentication tag for everything absorbed so far, treated as associated data
    /// (MAC mode: message is AD, mlen = 0). Port of `State128L.mac`.
    fn mac_128(&mut self, adlen: usize, mlen: usize) -> [u8; 16] {
        let mut sizes = [0u8; 16];
        sizes[0..8].copy_from_slice(&((adlen as u64) * 8).to_le_bytes());
        sizes[8..16].copy_from_slice(&((mlen as u64) * 8).to_le_bytes());

        let tmp = xor_blocks(&sizes, &self.blocks[2]);
        for _ in 0..7 {
            self.update(&tmp, &tmp);
        }

        let mut tag = self.blocks[0];
        for block in &self.blocks[1..7] {
            tag = xor_blocks(&tag, block);
        }
        tag
    }
}

// ---------------------------------------------------------------------------
// MAC wrapper. Port of `AegisMacType(Aegis128L)` (a.k.a. Aegis128LMac_128) and of
// `ChecksumStream` from src/vsr/checksum.zig.
// ---------------------------------------------------------------------------

/// Streaming checksum state. Mirrors upstream `vsr.ChecksumStream`.
pub struct ChecksumStream {
    state: State128L,
    buf: [u8; BLOCK_LENGTH],
    off: usize,
    msg_len: usize,
}

const BLOCK_LENGTH: usize = 32;

impl ChecksumStream {
    #[must_use]
    pub fn new() -> Self {
        // DEVIATION: upstream lazily initializes a global seed state under std.once(); here the
        // seed state is a pure function of the all-zero key, so we construct it directly.
        let key = [0u8; 16];
        let nonce = [0u8; 16];
        Self { state: State128L::new(&key, &nonce), buf: [0u8; BLOCK_LENGTH], off: 0, msg_len: 0 }
    }

    pub fn add(&mut self, bytes: &[u8]) {
        self.msg_len += bytes.len();

        let len_partial = ::core::cmp::min(bytes.len(), BLOCK_LENGTH - self.off);
        self.buf[self.off..][..len_partial].copy_from_slice(&bytes[..len_partial]);
        self.off += len_partial;
        if self.off < BLOCK_LENGTH {
            return;
        }
        let buf = self.buf;
        self.state.absorb(&buf);

        let mut i = len_partial;
        self.off = 0;
        while i + BLOCK_LENGTH <= bytes.len() {
            let mut chunk = [0u8; BLOCK_LENGTH];
            chunk.copy_from_slice(&bytes[i..i + BLOCK_LENGTH]);
            self.state.absorb(&chunk);
            i += BLOCK_LENGTH;
        }
        if i != bytes.len() {
            self.off = bytes.len() - i;
            self.buf[..self.off].copy_from_slice(&bytes[i..]);
        }
    }

    /// Returns the 128-bit checksum and consumes the stream (upstream invalidates the stream
    /// after finalizing as well).
    #[must_use]
    pub fn checksum(mut self) -> u128 {
        if self.off > 0 {
            let mut pad = [0u8; BLOCK_LENGTH];
            pad[..self.off].copy_from_slice(&self.buf[..self.off]);
            self.state.absorb(&pad);
        }
        let tag = self.state.mac_128(self.msg_len, 0);
        u128::from_le_bytes(tag)
    }
}

impl Default for ChecksumStream {
    fn default() -> Self {
        Self::new()
    }
}

/// The vsr checksum: AEGIS-128L MAC with zero key and zero nonce, message treated as AD.
#[must_use]
pub fn checksum(source: &[u8]) -> u128 {
    let mut stream = ChecksumStream::new();
    stream.add(source);
    stream.checksum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdx::prng::Prng;

    /// Upstream: test "checksum empty".
    #[test]
    fn checksum_empty() {
        let stream = ChecksumStream::new();
        assert_eq!(stream.checksum(), checksum(&[]));
    }

    /// Upstream: test "checksum test vectors".
    ///
    /// Note: these test vectors are not independent --- there are test vectors in AEAD papers,
    /// but they don't zero all of (nonce, key, secret message). However, as the underlying AEAD
    /// implementation matches those test vectors, the entries here are correct.
    ///
    /// They can be used to smoke-test independent implementations of TigerBeetle checksum.
    ///
    /// "checksum stability" further nails down the exact behavior.
    #[test]
    fn checksum_test_vectors() {
        // Upstream stores these as @byteSwap(u128 literal) so that the hex digits read in tag
        // byte order; from_le_bytes of the raw tag bytes is the same value.
        const EMPTY_TAG: [u8; 16] = [
            0x83, 0xcc, 0x60, 0x0d, 0xc4, 0xe3, 0xe7, 0xe6, //
            0x2d, 0x40, 0x55, 0x82, 0x61, 0x74, 0xf1, 0x49,
        ];
        const ZEROS_16_TAG: [u8; 16] = [
            0xf7, 0x2a, 0xd4, 0x8d, 0xd0, 0x5d, 0xd1, 0x65, //
            0x61, 0x33, 0x10, 0x1c, 0xd4, 0xbe, 0x3a, 0x26,
        ];

        assert_eq!(checksum(&[]), u128::from_le_bytes(EMPTY_TAG));
        assert_eq!(checksum(&[0x00; 16]), u128::from_le_bytes(ZEROS_16_TAG));
    }

    /// Upstream: test "checksum simple fuzzing".
    ///
    /// DEVIATION: upstream hashes up to 1 MiB x 1000 iterations on hardware AES in release mode;
    /// scaled down here so debug-mode software AES keeps tests fast. The pure-function and
    /// avalanche properties checked are unchanged.
    #[test]
    fn checksum_simple_fuzzing() {
        const MSG_MIN: usize = 1;
        const MSG_MAX: usize = 64 * crate::stdx::KIB;

        let mut prng = Prng::from_seed(42);
        let mut msg_buf = vec![0u8; MSG_MAX];

        for _ in 0..32 {
            let msg_len = prng.range_inclusive_usize(MSG_MIN, MSG_MAX);
            let msg = &mut msg_buf[..msg_len];
            prng.fill(msg);

            let msg_checksum = checksum(msg);

            // Sanity check that it's a pure function.
            let msg_checksum_again = checksum(msg);
            assert_eq!(msg_checksum, msg_checksum_again);

            // Change the message and make sure the checksum changes.
            let index = prng.index(msg.len());
            msg[index] = msg[index].wrapping_add(1);
            let changed_checksum = checksum(msg);
            assert_ne!(changed_checksum, msg_checksum);
        }
    }

    /// Upstream: test "checksum stability" --- change detector to ensure we don't inadvertently
    /// modify our checksum function.
    #[test]
    fn checksum_stability() {
        let mut buf = [0u8; 1024];
        let mut cases = [0u128; 896];
        let mut case_index = 0;

        // Zeros of various lengths.
        for subcase in 0..128 {
            let message = &mut buf[0..subcase];
            message.fill(0);

            cases[case_index] = checksum(message);
            case_index += 1;
        }

        // 64 bytes with exactly one bit set.
        for subcase in 0..64 * 8 {
            let message = &mut buf[0..64];
            message.fill(0);
            message[subcase / 8] = 1u8 << (subcase % 8);

            cases[case_index] = checksum(message);
            case_index += 1;
        }

        // Pseudo-random data from a specific PRNG of various lengths.
        let mut prng = Prng::from_seed(92);
        for subcase in 0..256 {
            let message = &mut buf[0..subcase + 13];
            prng.fill(message);

            cases[case_index] = checksum(message);
            case_index += 1;
        }
        assert_eq!(case_index, cases.len());

        // Sanity check that we are not getting trivial answers.
        for (i, case_a) in cases.iter().enumerate() {
            assert_ne!(*case_a, 0);
            assert_ne!(*case_a, u128::MAX);
            for case_b in &cases[0..i] {
                assert_ne!(case_a, case_b);
            }
        }

        // Hash me, baby, one more time! If this final hash changes, we broke compatibility in a
        // major way. (Upstream asserts little-endian at comptime; `to_le_bytes` pins that here.)
        let bytes: Vec<u8> = cases.iter().flat_map(|c| c.to_le_bytes()).collect();
        assert_eq!(checksum(&bytes), 0x82dc_aacf_4875_b279_4468_25b6_830d_1263);
    }

    /// Upstream: test "checksum alignment and sizing".
    ///
    /// DEVIATION: upstream allocates a deliberately misaligned input; our software round function
    /// is alignment-agnostic by construction (`&[u8]`), so the structural coverage is ported as-is.
    #[test]
    fn checksum_alignment_and_sizing() {
        const WINDOW_SIZE: usize = 256;

        let mut input = vec![0u8; 8 * crate::stdx::KIB];

        let mut prng = Prng::from_seed(92);
        prng.fill(&mut input);

        let mut cases = vec![0u128; 4112];
        let mut case_index = 0;

        for start_idx in 0..16 {
            cases[case_index] = checksum(&input[start_idx..]);
            case_index += 1;
            for size in 0..256 {
                cases[case_index] = checksum(&input[start_idx..][..size]);
                case_index += 1;
            }
        }
        assert_eq!(case_index, cases.len());

        for case in &cases {
            assert_ne!(*case, 0);
            assert_ne!(*case, u128::MAX);
        }

        for (idx, byte) in input.iter_mut().enumerate() {
            *byte = u8::from(idx % 2 == 1);
        }

        let even = checksum(&input[0..WINDOW_SIZE]);
        let odd = checksum(&input[1..][..WINDOW_SIZE]);

        for start_idx in 0..16usize {
            if start_idx.is_multiple_of(2) {
                assert_eq!(even, checksum(&input[start_idx..][..WINDOW_SIZE]));
            } else {
                assert_eq!(odd, checksum(&input[start_idx..][..WINDOW_SIZE]));
            }
        }

        let bytes: Vec<u8> = cases.iter().flat_map(|c| c.to_le_bytes()).collect();
        // Upstream compares against the native u128 literal 0xC8E7102D72CE96458639F6027DA0FBA0:
        assert_eq!(checksum(&bytes), 0xC8E7_102D_72CE_9645_8639_F602_7DA0_FBA0);
    }
}
