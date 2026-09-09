//! Physical and logical representation of IP addresses.
//!
//! Port of `src/stdx/net.zig`. Upstream stores an IPv6 or IPv6-mapped-IPv4 address as a
//! network-byte-order 16-byte array (so IPv4 and IPv6 share one uniform wire representation),
//! and rejects scopes/flowinfo/interface ids.
//!
//! DEVIATION: upstream declares `big: [16]u8 align(16)` and asserts little-endian hosts; the
//! alignment is not expressible in safe Rust and is irrelevant to parsing/formatting, so the
//! field here is a plain `[u8; 16]` network-byte-order array.

use core::fmt;

use crate::stdx::{ParseIntOptions, parse_int_u8, parse_int_u16};

/// Upstream `IPAddress.IPv4_prefix` (as a big-endian integer).
const IPV4_PREFIX: u128 = 0x0000_0000_0000_0000_0000_FFFF_0000_0000;

/// Upstream `IPAddress.IPv4_prefix_octets`: the first 12 bytes of the big-endian encoding of
/// `IPV4_PREFIX` — ten zero octets then `0xFF 0xFF`.
const IPV4_PREFIX_OCTETS: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF];

/// Error returned by [`IPAddress::parse`]; upstream `error.InvalidIPAddress`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidIpAddress;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Ipv4,
    Ipv6,
}

/// An IPv6 or IPv6-mapped IPv4 address (upstream `IPAddress`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IPAddress(pub [u8; 16]);

impl IPAddress {
    /// Upstream `IPAddress.from_v4`: map `[a,b,c,d]` into an IPv6-mapped address.
    #[must_use]
    pub const fn from_v4(octets: [u8; 4]) -> Self {
        Self([
            IPV4_PREFIX_OCTETS[0],
            IPV4_PREFIX_OCTETS[1],
            IPV4_PREFIX_OCTETS[2],
            IPV4_PREFIX_OCTETS[3],
            IPV4_PREFIX_OCTETS[4],
            IPV4_PREFIX_OCTETS[5],
            IPV4_PREFIX_OCTETS[6],
            IPV4_PREFIX_OCTETS[7],
            IPV4_PREFIX_OCTETS[8],
            IPV4_PREFIX_OCTETS[9],
            IPV4_PREFIX_OCTETS[10],
            IPV4_PREFIX_OCTETS[11],
            octets[0],
            octets[1],
            octets[2],
            octets[3],
        ])
    }

    /// Upstream `IPAddress.from_v6`.
    #[must_use]
    pub const fn from_v6(big: [u8; 16]) -> Self {
        Self(big)
    }

    /// Upstream `IPAddress.family`: an IPv6-mapped IPv4 address is IPv4.
    #[must_use]
    pub const fn family(self) -> Family {
        if self.as_u128() >> 32 == IPV4_PREFIX >> 32 { Family::Ipv4 } else { Family::Ipv6 }
    }

    /// The IPv4 octets if this is an IPv6-mapped IPv4 address (upstream `as_v4`).
    #[must_use]
    pub const fn as_v4(self) -> Option<[u8; 4]> {
        if matches!(self.family(), Family::Ipv6) {
            return None;
        }
        Some([self.0[12], self.0[13], self.0[14], self.0[15]])
    }

    /// Upstream `IPAddress.as_u128`: the 16 bytes read as a big-endian integer.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        u128::from_be_bytes(self.0)
    }

    /// Upstream `IPAddress.parse`: `'a.b.c.d'` selects IPv4, everything else is parsed as IPv6.
    ///
    /// # Errors
    /// Returns [`InvalidIpAddress`] when the text is not a well-formed address.
    pub fn parse(text: &str) -> Result<Self, InvalidIpAddress> {
        if text.contains('.') { Self::parse_v4(text) } else { Self::parse_v6(text) }
    }

    fn parse_v4(text: &str) -> Result<Self, InvalidIpAddress> {
        let mut octets = [0_u8; 4];
        let mut rest = text;
        for octet in &mut octets[..3] {
            let (head, tail) = rest.split_once('.').ok_or(InvalidIpAddress)?;
            *octet = Self::parse_octet(head)?;
            rest = tail;
        }
        octets[3] = Self::parse_octet(rest)?;
        Ok(Self::from_v4(octets))
    }

    fn parse_octet(text: &str) -> Result<u8, InvalidIpAddress> {
        parse_int_u8(
            text,
            ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: false },
        )
        .map_err(|_| InvalidIpAddress)
    }

    fn parse_v6(text: &str) -> Result<Self, InvalidIpAddress> {
        let (prefix, suffix) = match text.split_once("::") {
            Some((prefix, suffix)) => (prefix, Some(suffix)),
            None => (text, None),
        };

        for affix in [prefix, suffix.unwrap_or("")] {
            if affix.ends_with(':') || affix.starts_with(':') {
                return Err(InvalidIpAddress);
            }
        }

        let prefix_count = quibble_count(prefix);
        let suffix_count = quibble_count(suffix.unwrap_or(""));
        if prefix_count + suffix_count > 8 {
            return Err(InvalidIpAddress);
        }
        let shorthand_count = 8 - (prefix_count + suffix_count);
        match suffix {
            None => {
                if prefix_count != 8 {
                    return Err(InvalidIpAddress);
                }
            }
            Some(_) => {
                // `shorthand_count == 1` is non-canonical, but valid.
                if shorthand_count == 0 {
                    return Err(InvalidIpAddress);
                }
            }
        }

        let mut quibbles_big = [0_u16; 8];
        let mut index = 0;
        let mut rest = prefix;
        for _ in 0..prefix_count {
            let (quibble_text, rest_next) = rest.split_once(':').unwrap_or((rest, ""));
            rest = rest_next;
            quibbles_big[index] = Self::parse_quibble(quibble_text)?;
            index += 1;
        }
        if let Some(mut suffix_text) = suffix {
            index = (prefix_count + shorthand_count) as usize;
            for _ in 0..suffix_count {
                let (quibble_text, rest_next) =
                    suffix_text.split_once(':').unwrap_or((suffix_text, ""));
                suffix_text = rest_next;
                quibbles_big[index] = Self::parse_quibble(quibble_text)?;
                index += 1;
            }
        }

        let mut big = [0_u8; 16];
        for (i, quibble) in quibbles_big.iter().enumerate() {
            big[2 * i..2 * i + 2].copy_from_slice(&quibble.to_be_bytes());
        }
        Ok(Self(big))
    }

    fn parse_quibble(text: &str) -> Result<u16, InvalidIpAddress> {
        parse_int_u16(
            text,
            ParseIntOptions { base: 16, allow_leading_zero: true, allow_separators: false },
        )
        .map_err(|_| InvalidIpAddress)
    }
}

/// Upstream `IPAddress.quibble_count`.
fn quibble_count(text: &str) -> u32 {
    if text.is_empty() {
        0
    } else {
        let count = text.bytes().filter(|&c| c == b':').count() + 1;
        // The count only needs to be compared against the fixed quibble width (8), so saturating
        // at `u32::MAX` is equivalent to an error for any realistic input.
        u32::try_from(count).unwrap_or(u32::MAX)
    }
}

/// A run of zero quibbles to compress (RFC 5952 §4.2.1).
#[derive(Clone, Copy)]
struct Run {
    start: usize,
    count: usize,
}

/// Computes the run of zeros to compress: the first longest run, len ≥ 2.
///
/// DEVIATION: upstream uses a single `for` with a running carry; here the carry is tracked as
/// an `Option<Run>` with the same first-longest-tie semantics.
fn compressable_run(quibbles: &[u16; 8]) -> Option<Run> {
    let mut longest: Option<Run> = None;
    let mut current: Option<Run> = None;
    for (index, &quibble) in quibbles.iter().enumerate() {
        if quibble == 0 {
            current = Some(match current {
                Some(run) => Run { start: run.start, count: run.count + 1 },
                None => Run { start: index, count: 1 },
            });
            // `current` is guaranteed `Some` here (just assigned above); the `unwrap_or(0)`
            // fallback is never taken.
            let longest_count = longest.map(|run| run.count);
            let current_count = current.map(|run| run.count);
            if longest_count.is_none_or(|longest_count| longest_count < current_count.unwrap_or(0))
            {
                longest = current;
            }
        } else {
            current = None;
        }
    }
    let run = longest?;
    if run.count < 2 {
        return None;
    }
    Some(run)
}

impl fmt::Display for IPAddress {
    /// Upstream `IPAddress.format` — canonical RFC 5952 representation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.family() {
            Family::Ipv4 => {
                // `family()` returned IPv4, so `as_v4` cannot fail.
                let Some(octets) = self.as_v4() else { unreachable!() };
                write!(f, "{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
            }
            Family::Ipv6 => {
                let mut quibbles = [0_u16; 8];
                for (index, quibble) in quibbles.iter_mut().enumerate() {
                    *quibble = u16::from_be_bytes([self.0[2 * index], self.0[2 * index + 1]]);
                }
                write_ipv6(f, &quibbles)
            }
        }
    }
}

impl fmt::Debug for IPAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn write_ipv6(f: &mut fmt::Formatter<'_>, quibbles: &[u16; 8]) -> fmt::Result {
    let run = compressable_run(quibbles);
    let prefix_count = run.as_ref().map_or(8, |run: &Run| run.start);
    for (index, quibble) in quibbles[..prefix_count].iter().enumerate() {
        if index > 0 {
            f.write_str(":")?;
        }
        write!(f, "{quibble:x}")?;
    }
    if let Some(run) = run {
        f.write_str("::")?;
        for (offset, quibble) in quibbles[run.start + run.count..].iter().enumerate() {
            if offset > 0 {
                f.write_str(":")?;
            }
            write!(f, "{quibble:x}")?;
        }
    }
    Ok(())
}

/// An IP address + port (upstream `SocketAddress`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocketAddress {
    pub ip: IPAddress,
    pub port: u16,
}

impl fmt::Display for SocketAddress {
    /// IPv6 addresses are bracketed, matching `--addresses=[::1]:3000` syntax.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ip.family() {
            Family::Ipv4 => write!(f, "{}:{}", self.ip, self.port),
            Family::Ipv6 => write!(f, "[{}]:{}", self.ip, self.port),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn from_v4_packs_ipv4_mapped_prefix() {
        let v4 = IPAddress::from_v4([1, 2, 3, 4]);
        assert_eq!(v4.0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 4]);
        assert_eq!(v4.family(), Family::Ipv4);
        assert_eq!(v4.as_v4(), Some([1, 2, 3, 4]));
        assert_eq!(v4.to_string(), "1.2.3.4");
    }

    #[test]
    fn ipv4_parse_round_trip() {
        for (text, octets) in [
            ("127.0.0.1", [127, 0, 0, 1]),
            ("0.0.0.0", [0, 0, 0, 0]),
            ("255.255.255.255", [255, 255, 255, 255]),
        ] {
            let ip = IPAddress::parse(text).unwrap();
            assert_eq!(ip.family(), Family::Ipv4, "{text}");
            assert_eq!(ip.as_v4(), Some(octets), "{text}");
            assert_eq!(ip.to_string(), text);
        }
        for text in ["0.0.0.001", "256.0.0.1", "127.0.0.1.", ".127.0.0.1", "127.0.0"] {
            assert_eq!(IPAddress::parse(text), Err(InvalidIpAddress), "{text}");
        }
    }

    #[test]
    fn ipv6_parse_canonical_round_trip() {
        for text in [
            "::",
            "::1",
            "1::",
            "2001:db8::1:0:0:1",
            "2001:db8:0:1:1:1:1:1",
            "2001:db8::1",
            "ff01::101",
        ] {
            let ip = IPAddress::parse(text).unwrap();
            assert_eq!(ip.family(), Family::Ipv6, "{text}");
            assert_eq!(ip.to_string(), text, "{text}");
        }
    }

    #[test]
    fn ipv6_parse_accepts_noncanonical() {
        for (text, canonical) in [
            ("0::", "::"),
            ("2001:0db8:85a3::8a2e:0370:7334", "2001:db8:85a3::8a2e:370:7334"),
            ("2001:DB8:0:0:8:800:200C:417A", "2001:db8::8:800:200c:417a"),
            ("0:0:0:0:0:0:0:1", "::1"),
            ("0:0:0:0:0:0:0:0", "::"),
            ("Ff01::101", "ff01::101"),
        ] {
            let ip = IPAddress::parse(text).unwrap();
            assert_eq!(ip.to_string(), canonical, "{text}");
        }
    }

    #[test]
    fn ipv6_parse_rejects_invalid() {
        for text in ["", ":", ":::", "::::", "::d3:", ":1d7d::", "b::8%4", "::ffff:192.0.2.128"] {
            assert_eq!(IPAddress::parse(text), Err(InvalidIpAddress), "{text}");
        }
    }

    #[test]
    fn socket_address_display() {
        assert_eq!(
            SocketAddress { ip: IPAddress::from_v4([1, 2, 3, 4]), port: 3000 }.to_string(),
            "1.2.3.4:3000"
        );
        let loopback_v6 = IPAddress::from_v6([0; 16]).0;
        let mut loopback_v6 = loopback_v6;
        loopback_v6[15] = 1;
        assert_eq!(
            SocketAddress { ip: IPAddress::from_v6(loopback_v6), port: 3001 }.to_string(),
            "[::1]:3001"
        );
    }
}
