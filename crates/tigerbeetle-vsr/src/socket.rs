//! Cluster address parsing: `--addresses=...` and the `--statsd`/`amqp --host` socket form.
//!
//! Port of `src/vsr.zig` (rows 917-1026): `ClusterAddress`, `parse_addresses`,
//! `parse_address_and_port` and the local `parse_address` helper. The IP parsing itself lives in
//! `tigerbeetle_core::net`.

use tigerbeetle_core::constants::{ADDRESS, MEMBERS_MAX, PORT};
use tigerbeetle_core::net::{IPAddress, InvalidIpAddress, SocketAddress};
use tigerbeetle_core::stdx::parse_int_u16;

/// Upstream `vsr.ClusterAddress`: the parsed `--addresses` array plus a raw-string-checked
/// "magic zero" flag used only for testing.
///
/// DEVIATION: upstream stores a `BoundedArray` over `constants.members_max`; the max is
/// enforced at parse time here with a `Vec`, which needs no fixed upper bound on the struct.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClusterAddress {
    pub array: Vec<SocketAddress>,
    /// True when the value of `--addresses` is exactly `0` (enables "magic zero" mode for
    /// testing). The raw string is checked rather than the parsed address so the logic cannot
    /// be triggered by accident.
    pub zero: bool,
}

impl ClusterAddress {
    #[must_use]
    pub fn slice(&self) -> &[SocketAddress] {
        &self.array
    }

    #[must_use]
    pub fn members_count(&self) -> u8 {
        u8::try_from(self.array.len()).unwrap_or(u8::MAX)
    }

    /// Port of `ClusterAddress.parse_flag_value` — the `--addresses` flag parser.
    ///
    /// # Panics
    ///
    /// Panics when parsing yields an empty address list or a list longer than `MEMBERS_MAX`,
    /// matching upstream's assertions after an unchecked parse.
    ///
    /// # Errors
    ///
    /// Returns [`ParseAddressesError`] when `text` is not a valid address list.
    pub fn parse_flag_value(text: &str) -> Result<Self, ParseAddressesError> {
        let mut result = Self { array: Vec::new(), zero: text == "0" };
        result.array = parse_addresses(text)?;
        assert!(!result.array.is_empty());
        assert!(result.array.len() <= MEMBERS_MAX);
        Ok(result)
    }
}

/// Errors from [`parse_addresses`] / `ClusterAddress::parse_flag_value`, mirroring the
/// upstream `error` set so callers can reproduce upstream's diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseAddressesError {
    AddressHasTrailingComma,
    AddressLimitExceeded,
    AddressHasMoreThanOneColon,
    PortInvalid,
    AddressInvalid,
}

impl ParseAddressesError {
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::AddressHasTrailingComma => "invalid trailing comma:".to_owned(),
            Self::AddressLimitExceeded => {
                format!("too many addresses, at most {MEMBERS_MAX} are allowed:")
            }
            Self::AddressHasMoreThanOneColon => {
                "invalid address with more than one colon:".to_owned()
            }
            Self::PortInvalid => "invalid port:".to_owned(),
            Self::AddressInvalid => "invalid IPv4 or IPv6 address:".to_owned(),
        }
    }
}

/// Port of `vsr.parse_addresses`: split `raw` on commas, defaulting ports to
/// `constants.port`. The order is significant and is preserved.
///
/// # Panics
///
/// Panics when the number of parsed addresses disagrees with the comma count, matching
/// upstream's assertion.
///
/// # Errors
///
/// Returns [`ParseAddressesError`] on too many addresses, a trailing/empty address, or an
/// invalid address or port.
pub fn parse_addresses(raw: &str) -> Result<Vec<SocketAddress>, ParseAddressesError> {
    let address_count = raw.bytes().filter(|&c| c == b',').count() + 1;
    if address_count > MEMBERS_MAX {
        return Err(ParseAddressesError::AddressLimitExceeded);
    }

    let mut result = Vec::with_capacity(address_count);
    for raw_address in raw.split(',') {
        if raw_address.is_empty() {
            return Err(ParseAddressesError::AddressHasTrailingComma);
        }
        result.push(parse_address_and_port(raw_address, PORT)?);
    }
    assert_eq!(result.len(), address_count);
    Ok(result)
}

/// Port of `vsr.parse_address_and_port`: `host:port`, `host` (default port) or just `port`
/// (default address).
///
/// # Panics
///
/// Panics on an empty string or a zero `port_default`, matching upstream's assertions.
///
/// # Errors
///
/// Returns [`ParseAddressesError`] on an invalid host or port.
pub fn parse_address_and_port(
    string: &str,
    port_default: u16,
) -> Result<SocketAddress, ParseAddressesError> {
    assert!(!string.is_empty());
    assert!(port_default > 0);

    if let Some(split) = string.rfind([':', '.', ']']) {
        if string.as_bytes()[split] == b':' {
            let port = parse_int_u16(&string[split + 1..], parse_options::port())
                .map_err(|_| ParseAddressesError::PortInvalid)?;
            let ip = parse_address(&string[..split])?;
            return Ok(SocketAddress { ip, port });
        }
        let ip = parse_address(string)?;
        return Ok(SocketAddress { ip, port: port_default });
    }
    let Ok(ip) = IPAddress::parse(ADDRESS) else {
        unreachable!("constants.address must be a valid IP address")
    };
    let port = parse_int_u16(string, parse_options::port())
        .map_err(|_| ParseAddressesError::PortInvalid)?;
    Ok(SocketAddress { ip, port })
}

/// A variation of `stdx.IPAddress.parse` that requires `[]` around IPv6 addresses.
fn parse_address(string: &str) -> Result<IPAddress, ParseAddressesError> {
    if string.is_empty() {
        return Err(ParseAddressesError::AddressInvalid);
    }
    if string.ends_with(':') {
        return Err(ParseAddressesError::AddressHasMoreThanOneColon);
    }

    let expect_v6 = string.starts_with('[') && string.ends_with(']');
    let contains_colon = string.contains(':');
    if expect_v6 != contains_colon {
        return Err(ParseAddressesError::AddressInvalid);
    }

    let string_inner = if expect_v6 { &string[1..string.len() - 1] } else { string };
    IPAddress::parse(string_inner).map_err(invalid_ip)
}

fn invalid_ip(_: InvalidIpAddress) -> ParseAddressesError {
    ParseAddressesError::AddressInvalid
}

/// Parse options used by the port parser (upstream `stdx.parse_int(u16, .{})` defaults).
mod parse_options {
    use tigerbeetle_core::stdx::ParseIntOptions;

    pub(super) fn port() -> ParseIntOptions {
        ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: false }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use tigerbeetle_core::net::IPAddress;

    const fn v4(a: u8, b: u8, c: u8, d: u8) -> IPAddress {
        IPAddress::from_v4([a, b, c, d])
    }

    const fn v6_loopback() -> IPAddress {
        IPAddress::from_v6([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }

    const fn sock(ip: IPAddress, port: u16) -> SocketAddress {
        SocketAddress { ip, port }
    }

    /// Upstream `parse_addresses` positive vectors (addresses/ports, order preserved).
    #[test]
    fn parse_addresses_positive() {
        let cases: &[(&str, &[SocketAddress])] = &[
            (
                "1.2.3.4:567,0.0.0.0:0,255.255.255.255:65535",
                &[
                    sock(v4(1, 2, 3, 4), 567),
                    sock(v4(0, 0, 0, 0), 0),
                    sock(v4(255, 255, 255, 255), 65535),
                ],
            ),
            (
                "3.4.5.6:7777,200.3.4.5:6666,1.2.3.4:5555",
                &[
                    sock(v4(3, 4, 5, 6), 7777),
                    sock(v4(200, 3, 4, 5), 6666),
                    sock(v4(1, 2, 3, 4), 5555),
                ],
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(&parse_addresses(raw).unwrap(), expected, "{raw}");
        }
    }

    /// Default address and port apply to bare ports and bare hosts respectively.
    #[test]
    fn parse_addresses_defaults() {
        let default_ip = IPAddress::parse(ADDRESS).unwrap();
        let addresses = parse_addresses("1.2.3.4:5,4321,2.3.4.5").unwrap();
        assert_eq!(addresses.len(), 3);
        assert_eq!(addresses[0], sock(v4(1, 2, 3, 4), 5));
        assert_eq!(addresses[1], sock(default_ip, 4321));
        assert_eq!(addresses[2], sock(v4(2, 3, 4, 5), PORT));
    }

    /// Fewer addresses than the limit are fine.
    #[test]
    fn parse_addresses_less_than_limit() {
        let addresses = parse_addresses("1.2.3.4:5,4321").unwrap();
        assert_eq!(addresses.len(), 2);
    }

    /// IPv6 addresses require brackets; the default port applies when omitted.
    #[test]
    fn parse_addresses_ipv6() {
        let ip = v6_loopback();
        let addresses = parse_addresses("[::1]").unwrap();
        assert_eq!(addresses, vec![SocketAddress { ip, port: PORT }]);
        let addresses = parse_addresses("[::1]:3001,3000").unwrap();
        assert_eq!(addresses.len(), 2);
        assert_eq!(addresses[0], SocketAddress { ip, port: 3001 });
        // The bare "3000" parses against the default (IPv4) address.
        assert_eq!(
            addresses[1],
            SocketAddress { ip: IPAddress::parse(ADDRESS).unwrap(), port: 3000 }
        );
    }

    /// Negative vectors from upstream `parse_addresses`.
    #[test]
    fn parse_addresses_negative() {
        for (raw, expected) in [
            ("1.2.3.4,", ParseAddressesError::AddressHasTrailingComma),
            ("1.2.3.4,,5", ParseAddressesError::AddressHasTrailingComma),
            ("", ParseAddressesError::AddressHasTrailingComma),
            ("1.2.3.4:abc", ParseAddressesError::PortInvalid),
            ("1.2.3.4:65536", ParseAddressesError::PortInvalid),
            ("1.2.3.4: 5", ParseAddressesError::PortInvalid),
            ("1::2", ParseAddressesError::AddressHasMoreThanOneColon),
        ] {
            assert_eq!(parse_addresses(raw), Err(expected), "{raw}");
        }
    }

    /// `--addresses=0` is the magic-zero testing mode, determined from the raw string.
    #[test]
    fn cluster_address_zero_flag() {
        let address = ClusterAddress::parse_flag_value("0").unwrap();
        assert!(address.zero);
        // "0" is a valid port-less address on the default host, so parsing succeeds.
        assert_eq!(address.array.len(), 1);

        let address = ClusterAddress::parse_flag_value("127.0.0.1:3000").unwrap();
        assert!(!address.zero);
    }

    /// Upstream `parse_address_and_port` semantics via the shared helper.
    #[test]
    fn address_and_port_forms() {
        assert_eq!(
            parse_address_and_port("1.2.3.4:567", 3000).unwrap(),
            SocketAddress { ip: v4(1, 2, 3, 4), port: 567 }
        );
        assert_eq!(
            parse_address_and_port("2.3.4.5", 3000).unwrap(),
            SocketAddress { ip: v4(2, 3, 4, 5), port: 3000 }
        );
        // More than one colon with no brackets is a parse error (not "invalid address").
        assert_eq!(parse_address_and_port("1:2:3", 3000), Err(ParseAddressesError::AddressInvalid));
        assert_eq!(
            parse_address_and_port("1.2.3.4:5:6", 3000),
            Err(ParseAddressesError::AddressInvalid)
        );
    }
}
