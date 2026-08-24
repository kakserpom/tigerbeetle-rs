//! Upstream `stdx.hash_inline` / `low_level_hash` (`src/stdx/stdx.zig`).
//!
//! Inline version of Google Abseil `LowLevelHash` (inspired by `wyhash`):
//! <https://github.com/abseil-cpp/blob/master/absl/hash/internal/low_level_hash.cc>
//!
//! DEVIATION: upstream hashes the value's native-endian bytes; we always hash
//! little-endian so results are identical on every platform.

#![allow(clippy::cast_possible_truncation)] // hash arithmetic is defined by truncation

/// Hashes a byte slice with an optional seed, bit-for-bit compatible with upstream
/// `low_level_hash`.
#[must_use]
pub fn low_level_hash(seed: u64, input: &[u8]) -> u64 {
    const SALT: [u64; 5] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
    ];

    let starting_len = input.len();
    let mut state = seed ^ SALT[0];
    let mut input = input;

    if input.len() > 64 {
        let mut dup = [state, state];

        // Upstream consumes 64-byte blocks while *more* than 64 bytes remain, leaving the
        // final tail to the loops below.
        let blocks_end = ((input.len() - 1) / 64) * 64;
        for block in input[..blocks_end].as_chunks::<64>().0 {
            for (i, chunk) in [read_u64x4(block, 0), read_u64x4(block, 32)].iter().enumerate() {
                let mix1 = mul128(chunk[0] ^ SALT[i * 2 + 1], chunk[1] ^ dup[i]);
                let mix2 = mul128(chunk[2] ^ SALT[i * 2 + 2], chunk[3] ^ dup[i]);
                dup[i] = truncate64(mix1);
                dup[i] ^= truncate64(mix2);
            }
        }
        state = dup[0] ^ dup[1];
        input = &input[blocks_end..];
    }

    while input.len() > 16 {
        let mixed = mul128(read_u64(input, 0) ^ SALT[1], read_u64(input, 8) ^ state);
        state = truncate64(mixed);
        input = &input[16..];
    }

    let mut chunk = [0u64; 2];
    if input.len() > 8 {
        chunk[0] = read_u64(input, 0);
        chunk[1] = read_u64(input, input.len() - 8);
    } else if input.len() > 3 {
        chunk[0] = u64::from(u32_le(input));
        chunk[1] = u64::from(u32_le(&input[input.len() - 4..]));
    } else if !input.is_empty() {
        chunk[0] = (u64::from(input[0]) << 16)
            | (u64::from(input[input.len() / 2]) << 8)
            | u64::from(input[input.len() - 1]);
    }

    let mut mixed = mul128(chunk[0] ^ SALT[1], chunk[1] ^ state);
    // Truncate to 64 bits, then multiply by the length salt, as upstream does:
    mixed = u128::from(truncate64(mixed));
    let len_salt = (starting_len as u64) ^ SALT[1];
    mixed = mixed.wrapping_mul(u128::from(len_salt));
    truncate64(mixed)
}

fn read_u64x4(bytes: &[u8], offset: usize) -> [u64; 4] {
    [
        read_u64(bytes, offset),
        read_u64(bytes, offset + 8),
        read_u64(bytes, offset + 16),
        read_u64(bytes, offset + 24),
    ]
}

const fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut out = [0u8; 8];
    let mut i = 0;
    while i < 8 {
        out[i] = bytes[offset + i];
        i += 1;
    }
    u64::from_le_bytes(out)
}

const fn u32_le(bytes: &[u8]) -> u32 {
    let mut out = [0u8; 4];
    let mut i = 0;
    while i < 4 {
        out[i] = bytes[i];
        i += 1;
    }
    u32::from_le_bytes(out)
}

/// 64x64 -> 128 multiply (upstream widens operands to u128 and multiplies with wraparound).
fn mul128(a: u64, b: u64) -> u128 {
    (u128::from(a)) * (u128::from(b))
}

/// Upstream `@as(u64, @truncate(x ^ (x >> 64)))`.
const fn truncate64(x: u128) -> u64 {
    (x ^ (x >> 64)) as u64
}

/// Upstream `hash_inline(value)` specialized to `u64` (the only grid-cache use).
#[must_use]
pub fn hash_inline_u64(value: u64) -> u64 {
    low_level_hash(0, &value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{hash_inline_u64, low_level_hash};

    /// Upstream test vectors from `src/stdx/testing/low_level_hash_vectors.zig`
    /// (originally Abseil's `low_level_hash_test.cc`): `(seed, expected_hash, base64_input)`.
    const CASES: &[(u64, &str, &str)] = &[
        (0xeeee_0740_43a3_ee0f, "0000000000000000a6564b468248c683", "Zw=="),
        (0x0857_9020_89c3_93de, "0000000000000000ef192f401b116e1c", "xmk="),
        (0x993d_f040_024c_a3af, "0000000000000000be8dc0c54617639d", "c1H/"),
        (0xc4e4_c2ac_ea74_0e96, "000000000000000093d7f665b5521c8e", "SuwpzQ=="),
        (0x6a21_4b3d_b872_d0cf, "0000000000000000646d70bb42445f28", "uqvy++M="),
        (0x4434_3db6_a89d_ba4d, "000000000000000096a7b1e3cc9bd426", "RnzCVPgb"),
        (0x77b5_d6d1_ae1d_d483, "000000000000000076020289ab0790c4", "6OeNdlouYw=="),
        (0x89ab_8ecb_44d2_21f1, "000000000000000039f842e4133b9b44", "M5/JmmYyDbc="),
        (0x6024_4b17_577c_a81b, "00000000000000002b8d7047be4bcaab", "MVijWiVdBRdY"),
        (0x59a0_8dce_e071_7067, "000000000000000099628abef6716a97", "6V7Uq7LNxpu0VA=="),
        (0xf5f2_0db3_ade5_7396, "00000000000000004432e02ba42b2740", "EQ6CdEEhPdyHcOk="),
        (0xbf8d_ee07_51ad_3efb, "000000000000000074d810efcad7918a", "PqFB4fxnPgF+l+rc"),
        (0x6b7a_06b2_68d6_3e30, "000000000000000088c84e986002507f", "a5aPOFwq7LA7+zKvPA=="),
        (0xb8c3_7f0a_e0f5_4c82, "00000000000000004f99acf193cf39b9", "VOwY21wCGv5D+/qqOvs="),
        (0x9fcb_ed0c_38e5_0eef, "0000000000000000d90e7a3655891e37", "KdHmBTx8lHXYvmGJ+Vy7"),
        (0x2af4_bade_1d8e_3a1d, "00000000000000003bb378b1d4df8fcf", "qJkPlbHr8bMF7/cA6aE65Q=="),
        (0x714e_3aa9_12da_2f2c, "0000000000000000f78e94045c052d47", "ygvL0EhHZL0fIx6oHHtkxRQ="),
        (0xf5ee_75e3_cbb8_2c1c, "000000000000000026da0b2130da6b40", "c1rFXkt5YztwZCQRngncqtSs"),
        (0x620e_7007_321b_93b9, "000000000000000030b4d426af8c6986", "8hsQrzszzeNQSEcVXLtvIhm6mw=="),
        (0xc085_28ca_c2e5_51fc, "00000000000000005413b4aaf3baaeae", "ffUL4RocfyP4KfikGxO1yk7omDI="),
        (0x06a1_debf_9cc3_ad39, "0000000000000000756ab265370a1597", "OOB5TT00vF9Od/rLbAWshiErqhpV"),
        (
            0x7e0a_3c88_111f_c226,
            "0000000000000000daf5f4b7d09814fb",
            "or5wtXM7BFzTNpSzr+Lw5J5PMhVJ/Q==",
        ),
        (
            0x1301_fef1_5df3_9edb,
            "00000000000000008f874ae37742b75e",
            "gk6pCHDUsoopVEiaCrzVDhioRKxb844=",
        ),
        (
            0x064e_181f_3d58_17ab,
            "00000000000000008fecd03956121ce8",
            "TNctmwlC5QbEM6/No4R/La3UdkfeMhzs",
        ),
        (
            0xafaf_c449_6107_8ecb,
            "0000000000000000229c292ea7a08285",
            "SsQw9iAjhWz7sgcE9OwLuSC6hsM+BfHs2Q==",
        ),
        (
            0x4f7b_b455_4925_0094,
            "00000000000000000bb4bf0692d14bae",
            "ZzO3mVCj4xTT2TT3XqDyEKj2BZQBvrS8RHg=",
        ),
        (
            0x0a30_061a_baa2_818c,
            "0000000000000000207b24ca3bdac1db",
            "+klp5iPQGtppan5MflEls0iEUzqU+zGZkDJX",
        ),
        (
            0xd902_ee3e_44a5_705f,
            "000000000000000064f6cd6745d3825b",
            "RO6bvOnlJc8I9eniXlNgqtKy0IX6VNg16NRmgg==",
        ),
        (
            0x0316_d36d_a516_f583,
            "0000000000000000a2b2e1656b58df1e",
            "ZJjZqId1ZXBaij9igClE3nyliU5XWdNRrayGlYA=",
        ),
        (
            0x402d_83f9_f834_f616,
            "00000000000000000d01d30d9ee7a148",
            "7BfkhfGMDGbxfMB8uyL85GbaYQtjr2K8g7RpLzr/",
        ),
        (
            0x9c60_4164_c016_b72c,
            "00000000000000001cb4cd00ab804e3b",
            "rycWk6wHH7htETQtje9PidS2YzXBx+Qkg2fY7ZYS7A==",
        ),
        (
            0x3f45_07e0_1f9e_73ba,
            "00000000000000004697f2637fd90999",
            "RTkC2OUK+J13CdGllsH0H5WqgspsSa6QzRZouqx6pvI=",
        ),
        (
            0xc3fe_0d5b_e8d2_c7c7,
            "00000000000000008383a756b5688c07",
            "tKjKmbLCNyrLCM9hycOAXm4DKNpM12oZ7dLTmUx5iwAi",
        ),
        (
            0x5318_58a4_0bfa_7ea1,
            "0000000000000000695c29cb3696a975",
            "VprUGNH+5NnNRaORxgH/ySrZFQFDL+4VAodhfBNinmn8cg==",
        ),
        (
            0x8668_9478_a7a7_e8fa,
            "0000000000000000da2e5a5a5e971521",
            "gc1xZaY+q0nPcUvOOnWnT3bqfmT/geth/f7Dm2e/DemMfk4=",
        ),
        (
            0x4ec9_48b8_e7f2_7288,
            "00000000000000007935d4befa056b2b",
            "Mr35fIxqx1ukPAL0su1yFuzzAU3wABCLZ8+ZUFsXn47UmAph",
        ),
        (
            0x0ce4_6c72_13c1_0032,
            "000000000000000038dd541ca95420fe",
            "A9G8pw2+m7+rDtWYAdbl8tb2fT7FFo4hLi2vAsa5Y8mKH3CX3g==",
        ),
        (
            0xf63e_96ee_6f32_a8b6,
            "0000000000000000cc06c7a4963f967f",
            "DFaJGishGwEHDdj9ixbCoaTjz9KS0phLNWHVVdFsM93CvPft3hM=",
        ),
        (
            0x01cf_e85e_65fc_5225,
            "0000000000000000bf0f6f66e232fb20",
            "7+Ugx+Kr3aRNgYgcUxru62YkTDt5Hqis+2po81hGBkcrJg4N0uuy",
        ),
        (
            0x45c4_74f1_cee1_d2e8,
            "0000000000000000f7efb32d373fe71a",
            "H2w6O8BUKqu6Tvj2xxaecxEI2wRgIgqnTTG1WwOgDSINR13Nm4d4Vg==",
        ),
        (
            0x6e02_4e14_015f_329c,
            "0000000000000000e2e64634b1c12660",
            "1XBMnIbqD5jy65xTDaf6WtiwtdtQwv1dCVoqpeKj+7cTR1SaMWMyI04=",
        ),
        (
            0x760c_4050_2103_ae1c,
            "0000000000000000285b8fd1638e306d",
            "znZbdXG2TSFrKHEuJc83gPncYpzXGbAebUpP0XxzH0rpe8BaMQ17nDbt",
        ),
        (
            0x17fd_05c3_c560_c320,
            "0000000000000000658e8a4e3b714d6c",
            "ylu8Atu13j1StlcC1MRMJJXIl7USgDDS22HgVv0WQ8hx/8pNtaiKB17hCQ==",
        ),
        (
            0x8b34_200a_6f8e_90d9,
            "0000000000000000f391fb968e0eb398",
            "M6ZVVzsd7vAvbiACSYHioH/440dp4xG2mLlBnxgiqEvI/aIEGpD0Sf4VS0g=",
        ),
        (
            0x6be8_9e50_818b_df69,
            "0000000000000000744a9ea0cc144bf2",
            "li3oFSXLXI+ubUVGJ4blP6mNinGKLHWkvGruun85AhVn6iuMtocbZPVhqxzn",
        ),
        (
            0xfb38_9773_315b_47d8,
            "000000000000000012636f2be11012f1",
            "kFuQHuUCqBF3Tc3hO4dgdIp223ShaCoog48d5Do5zMqUXOh5XpGK1t5XtxnfGA==",
        ),
        (
            0x4f25_12a2_3f61_efee,
            "000000000000000029c57de825948f80",
            "jWmOad0v0QhXVJd1OdGuBZtDYYS8wBVHlvOeTQx9ZZnm8wLEItPMeihj72E0nWY=",
        ),
        (
            0x59cc_d92f_c16c_6fda,
            "000000000000000058c6f99ab0d1c021",
            "z+DHU52HaOQdW4JrZwDQAebEA6rm13Zg/9lPYA3txt3NjTBqFZlOMvTRnVzRbl23",
        ),
        (
            0x25c5_a7f5_bd33_0919,
            "000000000000000013e7b5a7b82fe3bb",
            "MmBiGDfYeTayyJa/tVycg+rN7f9mPDFaDc+23j0TlW9094er0ADigsl4QX7V3gG/qw==",
        ),
        (
            0x51df_4174_d34c_97d7,
            "000000000000000010fbc87901e02b63",
            "774RK+9rOL4iFvs1q2qpo/JVc/I39buvNjqEFDtDvyoB0FXxPI2vXqOrk08VPfIHkmU=",
        ),
        (
            0x080c_e6d7_6f89_cb57,
            "0000000000000000a24c9184901b748b",
            "+slatXiQ7/2lK0BkVUI1qzNxOOLP3I1iK6OfHaoxgqT63FpzbElwEXSwdsryq3UlHK0I",
        ),
        (
            0x2096_1c91_1965_f684,
            "0000000000000000cac4fd4c5080e581",
            "64mVTbQ47dHjHlOHGS/hjJwr/K2frCNpn87exOqMzNUVYiPKmhCbfS7vBUce5tO6Ec9osQ==",
        ),
        (
            0x4e5b_926e_c838_68e7,
            "0000000000000000c38bdb7483ba68e1",
            "fIsaG1r530SFrBqaDj1kqE0AJnvvK8MNEZbII2Yw1OK77v0V59xabIh0B5axaz/+a2V5WpA=",
        ),
        (
            0x3927_b30b_922e_ecef,
            "0000000000000000db2a8069b2ceaffa",
            "PGih0zDEOWCYGxuHGDFu9Ivbff/iE7BNUq65tycTR2R76TerrXALRosnzaNYO5fjFhTi+CiS",
        ),
        (
            0xbd02_9128_4a49_b61c,
            "0000000000000000df9fe91d0d1c7887",
            "RnpA/zJnEnnLjmICORByRVb9bCOgxF44p3VMiW10G7PvW7IhwsWajlP9kIwNA9FjAD2GoQHk2Q==",
        ),
        (
            0x073a_77c5_75bc_c956,
            "0000000000000000e83f49e96e2e6a08",
            "qFklMceaTHqJpy2qavJE+EVBiNFOi6OxjOA3LeIcBop1K7w8xQi3TrDk+BrWPRIbfprszSaPfrI=",
        ),
        (
            0x766a_0e2a_de6d_09a6,
            "00000000000000000c69e61b62ca2b62",
            "cLbfUtLl3EcQmITWoTskUR8da/VafRDYF/ylPYwk7/zazk6ssyrzxMN3mmSyvrXR2yDGNZ3WDrTT",
        ),
        (
            0x2599_f4f9_0511_5869,
            "0000000000000000b4a4f3f85f8298fe",
            "s/Jf1+FbsbCpXWPTUSeWyMH6e4CvTFvPE5Fs6Z8hvFITGyr0dtukHzkI84oviVLxhM1xMxrMAy1dbw==",
        ),
        (
            0xd825_6e54_44d2_1e53,
            "0000000000000000167a1b39e1e95f41",
            "FvyQ00+j7nmYZVQ8hI1Edxd0AWplhTfWuFGiu34AK5X8u2hLX1bE97sZM0CmeLe+7LgoUT1fJ/axybE=",
        ),
        (
            0xf664_a913_33fb_8dfd,
            "0000000000000000f8a2a5649855ee41",
            "L8ncxMaYLBH3g9buPu8hfpWZNlOF7nvWLNv9IozH07uQsIBWSKxoPy8+LW4tTuzC6CIWbRGRRD1sQV/4",
        ),
        (
            0x9625_b859_be37_2cd1,
            "000000000000000027992565b595c498",
            "CDK0meI07yrgV2kQlZZ+wuVqhc2NmzqeLH7bmcA6kchsRWFPeVF5Wqjjaj556ABeUoUr3yBmfU3kWOakkg==",
        ),
        (
            0x7b99_9407_82e2_9898,
            "00000000000000003e08cca5b71f9346",
            "d23/vc5ONh/HkMiq+gYk4gaCNYyuFKwUkvn46t+dfVcKfBTYykr4kdvAPNXGYLjM4u1YkAEFpJP+nX7eOvs=",
        ),
        (
            0x4fe1_2fa5_383b_51a8,
            "0000000000000000ad406b10c770a6d2",
            "NUR3SRxBkxTSbtQORJpu/GdR6b/h6sSGfsMj/KFd99ahbh+9r7LSgSGmkGVB/mGoT0pnMTQst7Lv2q6QN6Vm",
        ),
        (
            0xe2cc_b09a_c0f5_b4b6,
            "0000000000000000d1713ce6e552bcf2",
            "2BOFlcI3Z0RYDtS9T9Ie9yJoXlOdigpPeeT+CRujb/O39Ih5LPC9hP6RQk1kYESGyaLZZi3jtabHs7DiVx/VDg==",
        ),
        (
            0x7d0a_37ad_bd7b_753b,
            "0000000000000000753b287194c73ad3",
            "FF2HQE1FxEvWBpg6Z9zAMH+Zlqx8S1JD/wIlViL6ZDZY63alMDrxB0GJQahmAtjlm26RGLnjW7jmgQ4Ie3I+014=",
        ),
        (
            0xd3ae_96ef_9f71_85f2,
            "00000000000000005ae41a95f600af1c",
            "tHmO7mqVL/PX11nZrz50Hc+M17Poj5lpnqHkEN+4bpMx/YGbkrGOaYjoQjgmt1X2QyypK7xClFrjeWrCMdlVYtbW",
        ),
        (
            0x4fb8_8ea6_3f79_a0d8,
            "00000000000000004a61163b86a8bb4c",
            "/WiHi9IQcxRImsudkA/KOTqGe8/gXkhKIHkjddv5S9hi02M049dIK3EUyAEjkjpdGLUs+BN0QzPtZqjIYPOgwsYE9g==",
        ),
        (
            0xed56_4e25_9bb5_ebe9,
            "000000000000000042eeaa79e760c7e4",
            "qds+1ExSnU11L4fTSDz/QE90g4Jh6ioqSh3KDOTOAo2pQGL1k/9CCC7J23YF27dUTzrWsCQA2m4epXoCc3yPHb3xElA=",
        ),
        (
            0x3e32_56b6_0c42_8000,
            "0000000000000000698df622ef465b0a",
            "8FVYHx40lSQPTHheh08Oq0/pGm2OlG8BEf8ezvAxHuGGdgCkqpXIueJBF2mQJhTfDy5NncO8ntS7vaKs7sCNdDaNGOEi",
        ),
        (
            0x0fb0_5bad_59ec_8705,
            "0000000000000000157583111e1a6026",
            "4ZoEIrJtstiCkeew3oRzmyJHVt/pAs2pj0HgHFrBPztbQ10NsQ/lM6DM439QVxpznnBSiHMgMQJhER+70l72LqFTO1JiIQ==",
        ),
        (
            0xafdc_251d_bf97_b5f8,
            "0000000000000000aa1388f078e793e0",
            "hQPtaYI+wJyxXgwD5n8jGIKFKaFA/P83KqCKZfPthnjwdOFysqEOYwAaZuaaiv4cDyi9TyS8hk5cEbNP/jrI7q6pYGBLbsM=",
        ),
        (
            0x10ec_9c92_ddb5_dcbc,
            "0000000000000000f10d68d0f3309360",
            "S4gpMSKzMD7CWPsSfLeYyhSpfWOntyuVZdX1xSBjiGvsspwOZcxNKCRIOqAA0moUfOh3I5+juQV4rsqYElMD/gWfDGpsWZKQ",
        ),
        (
            0x9a76_7d58_22c7_dac4,
            "00000000000000002af056184457a3de",
            "oswxop+bthuDLT4j0PcoSKby4LhF47ZKg8K17xxHf74UsGCzTBbOz0MM8hQEGlyqDT1iUiAYnaPaUpL2mRK0rcIUYA4qLt5uOw==",
        ),
        (
            0xee46_2540_80d6_e2db,
            "00000000000000006d0058e1590b2489",
            "0II/697p+BtLSjxj5989OXI004TogEb94VUnDzOVSgMXie72cuYRvTFNIBgtXlKfkiUjeqVpd4a+n5bxNOD1TGrjQtzKU5r7obo=",
        ),
        (
            0xbbb6_6958_8d8b_f398,
            "0000000000000000638f287f68817f12",
            "E84YZW2qipAlMPmctrg7TKlwLZ68l4L+c0xRDUfyyFrA4MAti0q9sHq3TDFviH0Y+Kq3tEE5srWFA8LM9oomtmvm5PYxoaarWPLc",
        ),
        (
            0xdc2a_faa5_29be_ef44,
            "0000000000000000c46b71fecefd5467",
            "x3pa4HIElyZG0Nj7Vdy9IdJIR4izLmypXw5PCmZB5y68QQ4uRaVVi3UthsoJROvbjDJkP2DQ6L/eN8pFeLFzNPKBYzcmuMOb5Ull7w==",
        ),
        (
            0xf1f6_7391_d450_13a8,
            "00000000000000002c8e94679d964e0a",
            "jVDKGYIuWOP/QKLdd2wi8B2VJA8Wh0c8PwrXJVM8FOGM3voPDVPyDJOU6QsBDPseoR8uuKd19OZ/zAvSCB+zlf6upAsBlheUKgCfKww=",
        ),
        (
            0x16fc_e2b8_c65a_3429,
            "00000000000000008612b797ce22503a",
            "mkquunhmYe1aR2wmUz4vcvLEcKBoe6H+kjUok9VUn2+eTSkWs4oDDtJvNCWtY5efJwg/j4PgjRYWtqnrCkhaqJaEvkkOwVfgMIwF3e+d",
        ),
        (
            0xf4b0_9669_9f49_fe67,
            "000000000000000059f929babfba7170",
            "fRelvKYonTQ+s+rnnvQw+JzGfFoPixtna0vzcSjiDqX5s2Kg2//UGrK+AVCyMUhO98WoB1DDbrsOYSw2QzrcPe0+3ck9sePvb+Q/IRaHbw==",
        ),
        (
            0xca58_4c4b_c819_8682,
            "00000000000000009527556923fb49a0",
            "DUwXFJzagljo44QeJ7/6ZKw4QXV18lhkYT2jglMr8WB3CHUU4vdsytvw6AKv42ZcG6fRkZkq9fpnmXy6xG0aO3WPT1eHuyFirAlkW+zKtwg=",
        ),
        (
            0xed26_9fc3_818b_6aad,
            "00000000000000001039ab644f5e150b",
            "cYmZCrOOBBongNTr7e4nYn52uQUy2mfe48s50JXx2AZ6cRAt/xRHJ5QbEoEJOeOHsJyM4nbzwFm++SlT6gFZZHJpkXJ92JkR86uS/eV1hJUR",
        ),
        (
            0x33f2_53cb_b8fe_66a8,
            "00000000000000007816c83f3aa05e6d",
            "EXeHBDfhwzAKFhsMcH9+2RHwV+mJaN01+9oacF6vgm8mCXRd6jeN9U2oAb0of5c5cO4i+Vb/LlHZSMI490SnHU0bejhSCC2gsC5d2K30ER3iNA==",
        ),
        (
            0xd0b7_6b2c_1523_d99c,
            "0000000000000000f51d2f564518c619",
            "FzkzRYoNjkxFhZDso94IHRZaJUP61nFYrh5MwDwv9FNoJ5jyNCY/eazPZk+tbmzDyJIGw2h3GxaWZ9bSlsol/vK98SbkMKCQ/wbfrXRLcDzdd/8=",
        ),
        (
            0xfd28_f081_1a2a_237f,
            "000000000000000067d494cff03ac004",
            "Re4aXISCMlYY/XsX7zkIFR04ta03u4zkL9dVbLXMa/q6hlY/CImVIIYRN3VKP4pnd0AUr/ugkyt36JcstAInb4h9rpAGQ7GMVOgBniiMBZ/MGU7H",
        ),
        (
            0x0626_1fb1_3648_2e84,
            "00000000000000002802d636ced1cfbb",
            "ueLyMcqJXX+MhO4UApylCN9WlTQ+ltJmItgG7vFUtqs2qNwBMjmAvr5u0sAKd8jpzV0dDPTwchbIeAW5zbtkA2NABJV6hFM48ib4/J3A5mseA3cS8w==",
        ),
        (
            0x458e_fc75_0bca_7c3a,
            "0000000000000000f64e20bad771cb12",
            "6Si7Yi11L+jZMkwaN+GUuzXMrlvEqviEkGOilNq0h8TdQyYKuFXzkYc/q74gP3pVCyiwz9KpVGMM9vfnq36riMHRknkmhQutxLZs5fbmOgEO69HglCU=",
        ),
        (
            0xa7e6_9ff8_4e5e_7c27,
            "00000000000000000b9a6cf84a83e15e",
            "Q6AbOofGuTJOegPh9Clm/9crtUMQqylKrTc1fhfJo1tqvpXxhU4k08kntL1RG7woRnFrVh2UoMrL1kjin+s9CanT+y4hHwLqRranl9FjvxfVKm3yvg68",
        ),
        (
            0x3c59_bfd0_c29e_fe9e,
            "00000000000000008da6630319609301",
            "ieQEbIPvqY2YfIjHnqfJiO1/MIVRk0RoaG/WWi3kFrfIGiNLCczYoklgaecHMm/1sZ96AjO+a5stQfZbJQwS7Sc1ODABEdJKcTsxeW2hbh9A6CFzpowP1A==",
        ),
        (
            0x10be_facc_6afd_298d,
            "000000000000000040946a86e2a996f3",
            "zQUv8hFB3zh2GGl3KTvCmnfzE+SUgQPVaSVIELFX5H9cE3FuVFGmymkPQZJLAyzC90Cmi8GqYCvPqTuAAB//XTJxy4bCcVArgZG9zJXpjowpNBfr3ngWrSE=",
        ),
        (
            0x41d5_320b_0a38_efa7,
            "0000000000000000cab7f5997953fa76",
            "US4hcC1+op5JKGC7eIs8CUgInjKWKlvKQkapulxW262E/B2ye79QxOexf188u2mFwwe3WTISJHRZzS61IwljqAWAWoBAqkUnW8SHmIDwHUP31J0p5sGdP47L",
        ),
        (
            0x58db_1c74_50fe_17f3,
            "000000000000000039129ca0e04fc465",
            "9bHUWFna2LNaGF6fQLlkx1Hkt24nrkLE2CmFdWgTQV3FFbUe747SSqYw6ebpTa07MWSpWRPsHesVo2B9tqHbe7eQmqYebPDFnNqrhSdZwFm9arLQVs+7a3Ic6A==",
        ),
        (
            0x6098_c055_a335_b7a6,
            "00000000000000005238221fd685e1b8",
            "Kb3DpHRUPhtyqgs3RuXjzA08jGb59hjKTOeFt1qhoINfYyfTt2buKhD6YVffRCPsgK9SeqZqRPJSyaqsa0ovyq1WnWW8jI/NhvAkZTVHUrX2pC+cD3OPYT05Dag=",
        ),
        (
            0x1bba_cec6_7845_a801,
            "0000000000000000175130c407dbcaab",
            "gzxyMJIPlU+bJBwhFUCHSofZ/319LxqMoqnt3+L6h2U2+ZXJCSsYpE80xmR0Ta77Jq54o92SMH87HV8dGOaCTuAYF+lDL42SY1P316Cl0sZTS2ow3ZqwGbcPNs/1",
        ),
        (
            0x00c4_19cf_c744_2190,
            "000000000000000002f20e7536c0b0df",
            "uR7V0TW+FGVMpsifnaBAQ3IGlr1wx5sKd7TChuqRe6OvUXTlD4hKWy8S+8yyOw8lQabism19vOQxfmocEOW/vzY0pEa87qHrAZy4s9fH2Bltu8vaOIe+agYohhYORQ==",
        ),
        (
            0xc95e_510d_94ba_270c,
            "00000000000000002742cb488a04ad56",
            "1UR5eoo2aCwhacjZHaCh9bkOsITp6QunUxHQ2SfeHv0imHetzt/Z70mhyWZBalv6eAx+YfWKCUib2SHDtz/A2dc3hqUWX5VfAV7FQsghPUAtu6IiRatq4YSLpDvKZBQ=",
        ),
        (
            0xff1a_e05c_9808_9c3f,
            "0000000000000000d6afb593879ff93b",
            "opubR7H63BH7OtY+Avd7QyQ25UZ8kLBdFDsBTwZlY6gA/u+x+czC9AaZMgmQrUy15DH7YMGsvdXnviTtI4eVI4aF1H9Rl3NXMKZgwFOsdTfdcZeeHVRzBBKX8jUfh1il",
        ),
        (
            0x90c0_2b8d_cece_d493,
            "0000000000000000f50ad64caac0ca7f",
            "DC0kXcSXtfQ9FbSRwirIn5tgPri0sbzHSa78aDZVDUKCMaBGyFU6BmrulywYX8yzvwprdLsoOwTWN2wMjHlPDqrvVHNEjnmufRDblW+nSS+xtKNs3N5xsxXdv6JXDrAB/Q==",
        ),
        (
            0x9f8a_7669_7ab1_aa36,
            "00000000000000002ade95c4261364ae",
            "BXRBk+3wEP3Lpm1y75wjoz+PgB0AMzLe8tQ1AYU2/oqrQB2YMC6W+9QDbcOfkGbeH+b7IBkt/gwCMw2HaQsRFEsurXtcQ3YwRuPz5XNaw5NAvrNa67Fm7eRzdE1+hWLKtA8=",
        ),
        (
            0x6ba1_bf3d_811a_531d,
            "00000000000000005c4f3299faacd07a",
            "RRBSvEGYnzR9E45Aps/+WSnpCo/X7gJLO4DRnUqFrJCV/kzWlusLE/6ZU6RoUf2ROwcgEvUiXTGjLs7ts3t9SXnJHxC1KiOzxHdYLMhVvgNd3hVSAXODpKFSkVXND55G2L1W",
        ),
        (
            0x6a41_8974_109c_67b4,
            "0000000000000000fffe3bff0ae5e9bc",
            "jeh6Qazxmdi57pa9S3XSnnZFIRrnc6s8QLrah5OX3SB/V2ErSPoEAumavzQPkdKF1/SfvmdL+qgF1C+Yawy562QaFqwVGq7+tW0yxP8FStb56ZRgNI4IOmI30s1Ei7iops9Uuw==",
        ),
        (
            0x8472_f1c2_b3d2_30a3,
            "00000000000000001db785c0005166e4",
            "6QO5nnDrY2/wrUXpltlKy2dSBcmK15fOY092CR7KxAjNfaY+aAmtWbbzQk3MjBg03x39afSUN1fkrWACdyQKRaGxgwq6MGNxI6W+8DLWJBHzIXrntrE/ml6fnNXEpxplWJ1vEs4=",
        ),
        (
            0x5e06_068f_884e_73a7,
            "0000000000000000ea000d962ad18418",
            "0oPxeEHhqhcFuwonNfLd5jF3RNATGZS6NPoS0WklnzyokbTqcl4BeBkMn07+fDQv83j/BpGUwcWO05f3+DYzocfnizpFjLJemFGsls3gxcBYxcbqWYev51tG3lN9EvRE+X9+Pwww",
        ),
        (
            0x5529_0b1a_8f17_0f59,
            "0000000000000000e42aef38359362d9",
            "naSBSjtOKgAOg8XVbR5cHAW3Y+QL4Pb/JO9/oy6L08wvVRZqo0BrssMwhzBP401Um7A4ppAupbQeJFdMrysY34AuSSNvtNUy5VxjNECwiNtgwYHw7yakDUv8WvonctmnoSPKENegQg==",
        ),
        (
            0x5501_cfd8_3dfe_706a,
            "0000000000000000c8e95657348a3891",
            "vPyl8DxVeRe1OpilKb9KNwpGkQRtA94UpAHetNh+95V7nIW38v7PpzhnTWIml5kw3So1Si0TXtIUPIbsu32BNhoH7QwFvLM+JACgSpc5e3RjsL6Qwxxi11npwxRmRUqATDeMUfRAjxg=",
        ),
        (
            0xe43e_d13d_13a6_6990,
            "0000000000000000c162eca864f238c6",
            "QC9i2GjdTMuNC1xQJ74ngKfrlA4w3o58FhvNCltdIpuMhHP1YsDA78scQPLbZ3OCUgeQguYf/vw6zAaVKSgwtaykqg5ka/4vhz4hYqWU5ficdXqClHl+zkWEY26slCNYOM5nnDlly8Cj",
        ),
        (
            0xdf43_bc37_5cf5_283f,
            "0000000000000000be1fb373e20579ad",
            "7CNIgQhAHX27nxI0HeB5oUTnTdgKpRDYDKwRcXfSFGP1XeT9nQF6WKCMjL1tBV6x7KuJ91GZz11F4c+8s+MfqEAEpd4FHzamrMNjGcjCyrVtU6y+7HscMVzr7Q/ODLcPEFztFnwjvCjmHw==",
        ),
        (
            0x8112_b806_d288_d7b5,
            "0000000000000000628a1d4f40aa6ffd",
            "Qa/hC2RPXhANSospe+gUaPfjdK/yhQvfm4cCV6/pdvCYWPv8p1kMtKOX3h5/8oZ31fsmx4Axphu5qXJokuhZKkBUJueuMpxRyXpwSWz2wELx5glxF7CM0Fn+OevnkhUn5jsPlG2r5jYlVn8=",
        ),
        (
            0xd52a_18ab_b001_cb46,
            "0000000000000000a87bdb7456340f90",
            "kUw/0z4l3a89jTwN5jpG0SHY5km/IVhTjgM5xCiPRLncg40aqWrJ5vcF891AOq5hEpSq0bUCJUMFXgct7kvnys905HjerV7Vs1Gy84tgVJ70/2+pAZTsB/PzNOE/G6sOj4+GbTzkQu819OLB",
        ),
        (
            0xe12b_76a2_433a_1236,
            "00000000000000005960ef3ba982c801",
            "VDdfSDbO8Tdj3T5W0XM3EI7iHh5xpIutiM6dvcJ/fhe23V/srFEkDy5iZf/VnA9kfi2C79ENnFnbOReeuZW1b3MUXB9lgC6U4pOTuC+jHK3Qnpyiqzj7h3ISJSuo2pob7vY6VHZo6Fn7exEqHg==",
        ),
        (
            0x175b_f731_9cf1_fa00,
            "00000000000000005026586df9a431ec",
            "Ldfvy3ORdquM/R2fIkhH/ONi69mcP1AEJ6n/oropwecAsLJzQSgezSY8bEiEs0VnFTBBsW+RtZY6tDj03fnb3amNUOq1b7jbqyQkL9hpl+2Z2J8IaVSeownWl+bQcsR5/xRktIMckC5AtF4YHfU=",
        ),
        (
            0xd63d_57b3_f675_25ae,
            "0000000000000000fe4b8a20fdf0840b",
            "BrbNpb42+VzZAjJw6QLirXzhweCVRfwlczzZ0VX2xluskwBqyfnGovz5EuX79JJ31VNXa5hTkAyQat3lYKRADTdAdwE5PqM1N7YaMqqsqoAAAeuYVXuk5eWCykYmClNdSspegwgCuT+403JigBzi",
        ),
        (
            0x933f_aea8_5883_2b73,
            "0000000000000000dcb761867da7072f",
            "gB3NGHJJvVcuPyF0ZSvHwnWSIfmaI7La24VMPQVoIIWF7Z74NltPZZpx2f+cocESM+ILzQW9p+BC8x5IWz7N4Str2WLGKMdgmaBfNkEhSHQDU0IJEOnpUt0HmjhFaBlx0/LTmhua+rQ6Wup8ezLwfg==",
        ),
        (
            0x53d0_61e5_f8e7_c04f,
            "0000000000000000c10d4653667275b7",
            "hTKHlRxx6Pl4gjG+6ksvvj0CWFicUg3WrPdSJypDpq91LUWRni2KF6+81ZoHBFhEBrCdogKqeK+hy9bLDnx7g6rAFUjtn1+cWzQ2YjiOpz4+ROBB7lnwjyTGWzJD1rXtlso1g2qVH8XJVigC5M9AIxM=",
        ),
        (
            0xdb41_2455_6dd5_15e0,
            "0000000000000000727720deec13110b",
            "IWQBelSQnhrr0F3BhUpXUIDauhX6f95Qp+A0diFXiUK7irwPG1oqBiqHyK/SH/9S+rln9DlFROAmeFdH0OCJi2tFm4afxYzJTFR4HnR4cG4x12JqHaZLQx6iiu6CE3rtWBVz99oAwCZUOEXIsLU24o2Y",
        ),
        (
            0x4fb3_1a0d_d681_ee71,
            "0000000000000000710b009662858dc9",
            "TKo+l+1dOXdLvIrFqeLaHdm0HZnbcdEgOoLVcGRiCbAMR0j5pIFw8D36tefckAS1RCFOH5IgP8yiFT0Gd0a2hI3+fTKA7iK96NekxWeoeqzJyctc6QsoiyBlkZerRxs5RplrxoeNg29kKDTM0K94mnhD9g==",
        ),
        (
            0x27cc_72ee_fa13_8e4c,
            "0000000000000000fbf8f7a3ecac1eb7",
            "YU4e7G6EfQYvxCFoCrrT0EFgVLHFfOWRTJQJ5gxM3G2b+1kJf9YPrpsxF6Xr6nYtS8reEEbDoZJYqnlk9lXSkVArm88Cqn6d25VCx3+49MqC0trIlXtb7SXUUhwpJK16T0hJUfPH7s5cMZXc6YmmbFuBNPE=",
        ),
        (
            0x44bc_2dfb_a4bd_3ced,
            "0000000000000000b6fc4fcd0722e3df",
            "/I/eImMwPo1U6wekNFD1Jxjk9XQVi1D+FPdqcHifYXQuP5aScNQfxMAmaPR2XhuOQhADV5tTVbBKwCDCX4E3jcDNHzCiPvViZF1W27txaf2BbFQdwKrNCmrtzcluBFYu0XZfc7RU1RmxK/RtnF1qHsq/O4pp",
        ),
        (
            0x242d_a1e3_a439_bed8,
            "00000000000000007cb86dcc55104aac",
            "CJTT9WGcY2XykTdo8KodRIA29qsqY0iHzWZRjKHb9alwyJ7RZAE3V5Juv4MY3MeYEr1EPCCMxO7yFXqT8XA8YTjaMp3bafRt17Pw8JC4iKJ1zN+WWKOESrj+3aluGQqn8z1EzqY4PH7rLG575PYeWsP98BugdA==",
        ),
        (
            0xdc55_9c74_6e35_c139,
            "000000000000000019e71e9b45c3a51e",
            "ZlhyQwLhXQyIUEnMH/AEW27vh9xrbNKJxpWGtrEmKhd+nFqAfbeNBQjW0SfG1YI0xQkQMHXjuTt4P/EpZRtA47ibZDVS8TtaxwyBjuIDwqcN09eCtpC+Ls+vWDTLmBeDM3u4hmzz4DQAYsLiZYSJcldg9Q3wszw=",
        ),
        (
            0x0d0b_0350_275b_9989,
            "000000000000000051de38573c2bea48",
            "v2KU8y0sCrBghmnm8lzGJlwo6D6ObccAxCf10heoDtYLosk4ztTpLlpSFEyu23MLA1tJkcgRko04h19QMG0mOw/wc93EXAweriBqXfvdaP85sZABwiKO+6rtS9pacRVpYYhHJeVTQ5NzrvBvi1huxAr+xswhVMfL",
        ),
        (
            0xb044_89e4_1d17_730c,
            "0000000000000000a73ab6996d6df158",
            "QhKlnIS6BuVCTQsnoE67E/yrgogE8EwO7xLaEGei26m0gEU4OksefJgppDh3X0x0Cs78Dr9IHK5b977CmZlrTRmwhlP8pM+UzXPNRNIZuN3ntOum/QhUWP8SGpirheXENWsXMQ/nxtxakyEtrNkKk471Oov9juP8oQ==",
        ),
        (
            0x2217_285e_b457_2156,
            "000000000000000055ef2b8c930817b2",
            "/ZRMgnoRt+Uo6fUPr9FqQvKX7syhgVqWu+WUSsiQ68UlN0efSP6Eced5gJZL6tg9gcYJIkhjuQNITU0Q3TjVAnAcobgbJikCn6qZ6pRxKBY4MTiAlfGD3T7R7hwJwx554MAy++Zb/YUFlnCaCJiwQMnowF7aQzwYFCo=",
        ),
        (
            0x12c2_e8e6_8aed_e73b,
            "0000000000000000b2850bf5fae87157",
            "NB7tU5fNE8nI+SXGfipc7sRkhnSkUF1krjeo6k+8FITaAtdyz+o7mONgXmGLulBPH9bEwyYhKNVY0L+njNQrZ9YC2aXsFD3PdZsxAFaBT3VXEzh+NGBTjDASNL3mXyS8Yv1iThGfHoY7T4aR0NYGJ+k+pR6f+KrPC96M",
        ),
        (
            0x4d61_2125_bdc4_fd00,
            "0000000000000000ecf3de1acd04651f",
            "8T6wrqCtEO6/rwxF6lvMeyuigVOLwPipX/FULvwyu+1wa5sQGav/2FsLHUVn6cGSi0LlFwLewGHPFJDLR0u4t7ZUyM//x6da0sWgOa5hzDqjsVGmjxEHXiaXKW3i4iSZNuxoNbMQkIbVML+DkYu9ND0O2swg4itGeVSzXA==",
        ),
        (
            0x8182_6b55_3954_464e,
            "0000000000000000cc0a40552559ff32",
            "Ntf1bMRdondtMv1CYr3G80iDJ4WSAlKy5H34XdGruQiCrnRGDBa+eUi7vKp4gp3BBcVGl8eYSasVQQjn7MLvb3BjtXx6c/bCL7JtpzQKaDnPr9GWRxpBXVxKREgMM7d8lm35EODv0w+hQLfVSh8OGs7fsBb68nNWPLeeSOo=",
        ),
        (
            0xc2e5_d345_dc0d_dd2d,
            "0000000000000000c385c374f20315b1",
            "VsSAw72Ro6xks02kaiLuiTEIWBC5bgqr4WDnmP8vglXzAhixk7td926rm9jNimL+kroPSygZ9gl63aF5DCPOACXmsbmhDrAQuUzoh9ZKhWgElLQsrqo1KIjWoZT5b5QfVUXY9lSIBg3U75SqORoTPq7HalxxoIT5diWOcJQi",
        ),
        (
            0x3da6_830a_9e32_631e,
            "0000000000000000b90208a4c7234183",
            "j+loZ+C87+bJxNVebg94gU0mSLeDulcHs84tQT7BZM2rzDSLiCNxUedHr1ZWJ9ejTiBa0dqy2I2ABc++xzOLcv+//YfibtjKtYggC6/3rv0XCc7xu6d/O6xO+XOBhOWAQ+IHJVHf7wZnDxIXB8AUHsnjEISKj7823biqXjyP3g==",
        ),
        (
            0xc9ae_5c87_59b4_877a,
            "000000000000000058aa1ca7a4c075d9",
            "f3LlpcPElMkspNtDq5xXyWU62erEaKn7RWKlo540gR6mZsNpK1czV/sOmqaq8XAQLEn68LKj6/cFkJukxRzCa4OF1a7cCAXYFp9+wZDu0bw4y63qbpjhdCl8GO6Z2lkcXy7KOzbPE01ukg7+gN+7uKpoohgAhIwpAKQXmX5xtd0=",
        ),
    ];

    fn decode_base64(input: &str) -> Vec<u8> {
        fn value_of(byte: u8) -> u8 {
            match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => unreachable!("invalid base64 character {byte}"),
            }
        }

        let input = input.as_bytes();
        assert!(input.len().is_multiple_of(4));
        let mut out = Vec::with_capacity(input.len() / 4 * 3);
        for quad in input.as_chunks::<4>().0 {
            let mut padding = 0;
            for &byte in quad {
                if byte == b'=' {
                    padding += 1;
                }
            }
            assert!(padding <= 2);
            let mut values = [0u32; 4];
            for (slot, &byte) in quad.iter().enumerate() {
                values[slot] = u32::from(if byte == b'=' { 0 } else { value_of(byte) });
            }
            let triple = (values[0] << 18) | (values[1] << 12) | (values[2] << 6) | values[3];
            out.extend_from_slice(&[(triple >> 16) as u8, (triple >> 8) as u8, triple as u8]);
            out.truncate(out.len() - padding);
        }
        out
    }

    #[test]
    fn upstream_low_level_hash_vectors() {
        for &(seed, hash, b64) in CASES {
            let expected =
                u64::from_str_radix(hash, 16).unwrap_or_else(|_| panic!("bad hash literal {hash}"));
            let input = decode_base64(b64);
            assert_eq!(low_level_hash(seed, &input), expected, "input={b64:?}");
        }
    }

    #[test]
    fn hash_inline_u64_matches_low_level_hash_of_le_bytes() {
        for value in [0u64, 1, 42, u64::MAX, 0xdead_beef_cafe_f00d] {
            assert_eq!(hash_inline_u64(value), low_level_hash(0, &value.to_le_bytes()));
        }
    }
}
