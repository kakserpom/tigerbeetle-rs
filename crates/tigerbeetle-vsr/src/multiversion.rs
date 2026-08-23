//! Port of the `Release`/`ReleaseTriple` subset of `src/multiversion.zig`.
//!
//! The rest of multiversion (release lists, upgrade orchestration) lands later.
//! TODO(port): src/multiversion.zig Multiversion, ReleaseList, release-execute plumbing.

/// A semantic version triple packed into a `u32`, in on-disk/wire byte order
/// (little-endian: patch, minor, major).
///
/// Port of `multiversion.Release`. Upstream reinterprets the bytes of a
/// `ReleaseTriple { patch: u8, minor: u8, major: u16 }` extern struct as a `u32`; we compose the
/// value explicitly, which yields identical bytes on little-endian targets.
///
/// DEVIATION: no `unsafe`/byte-punning — see rule 3 of AGENTS.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Release {
    pub value: u32,
}

impl Release {
    pub const ZERO: Self = Self::from_triple(ReleaseTriple { major: 0, minor: 0, patch: 0 });
    /// Minimum is used for all development builds, to distinguish them from production deployments.
    pub const MINIMUM: Self = Self::from_triple(ReleaseTriple { major: 0, minor: 0, patch: 1 });

    /// The 65535.x.x releases are reserved for cluster=0.
    /// This way, when testing multiversion binaries (either manually or with the integration
    /// tests' or Vortex's build) it isn't possible to use the test's multiversion build to upgrade
    /// a production cluster to non-production code.
    pub const DEVELOPMENT_MAJOR: u16 = u16::MAX;

    #[must_use]
    pub const fn from_triple(release_triple: ReleaseTriple) -> Self {
        // Matches the little-endian layout of upstream's extern struct:
        // bytes = [patch, minor, major_lo, major_hi].
        let major = release_triple.major as u32;
        let minor = release_triple.minor as u32;
        let patch = release_triple.patch as u32;
        Self { value: (major << 16) | (minor << 8) | patch }
    }

    #[must_use]
    pub const fn triple(&self) -> ReleaseTriple {
        // Bit extraction by construction: low byte = patch, next byte = minor, high half = major.
        #[allow(clippy::cast_possible_truncation)]
        ReleaseTriple {
            major: (self.value >> 16) as u16,
            minor: (self.value >> 8) as u8,
            patch: self.value as u8,
        }
    }

    /// Test-only release binaries will only start when cluster=0.
    /// This ensures that test/development multiversion binaries can be tested by upgrading from
    /// actual production binaries, without risking accidentally upgrading a production cluster to a
    /// test/development binary.
    #[must_use]
    pub fn testing(&self) -> bool {
        self.triple().major == Self::DEVELOPMENT_MAJOR
    }

    #[must_use]
    pub fn max(a: Self, b: Self) -> Self {
        if a.value > b.value { a } else { b }
    }
}

/// Port of `multiversion.ReleaseTriple`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleaseTriple {
    pub major: u16,
    pub minor: u8,
    pub patch: u8,
}

impl ReleaseTriple {
    /// Port of `ReleaseTriple.parse`. Unlike upstream (which returns error codes through Zig's
    /// error unions and reports a static diagnostic via Flags), this simply returns `None` on any
    /// parse failure. TODO(port): diagnostics once the Flags/CLI layer exists.
    #[must_use]
    pub fn parse(string: &str) -> Option<Self> {
        let mut parts = string.split('.');
        let major = parts.next()?;
        let minor = parts.next()?;
        let patch = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Self { major: parse_int(major)?, minor: parse_int(minor)?, patch: parse_int(patch)? })
    }
}

fn parse_int<T>(s: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    s.parse::<T>().ok()
}

const _: () = assert!(size_of::<Release>() == 4);
const _: () = assert!(size_of::<ReleaseTriple>() == 4);

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream relies on byte-reinterpretation between Release and ReleaseTriple; pin our
    /// explicit composition to those exact bytes.
    #[test]
    fn release_triple_round_trip() {
        for &(major, minor, patch) in &[
            (0u16, 0u8, 0u8),
            (0, 0, 1),
            (65535, 255, 255),
            (1, 2, 3),
            (Release::DEVELOPMENT_MAJOR, 7, 9),
        ] {
            let triple = ReleaseTriple { major, minor, patch };
            let release = Release::from_triple(triple);
            assert_eq!(release.triple(), triple);
        }

        // Byte-level check against the upstream extern-struct memory layout (LE):
        assert_eq!(
            Release::from_triple(ReleaseTriple { major: 1, minor: 2, patch: 3 }).value,
            0x0001_0203
        );
        assert_eq!(
            Release::from_triple(ReleaseTriple {
                major: Release::DEVELOPMENT_MAJOR,
                minor: 0,
                patch: 0,
            })
            .value,
            0xffff_0000
        );
    }

    /// Upstream: test "ReleaseTriple.parse".
    #[test]
    fn release_triple_parse() {
        assert_eq!(
            ReleaseTriple::parse("1.2.3"),
            Some(ReleaseTriple { major: 1, minor: 2, patch: 3 })
        );
        assert_eq!(
            ReleaseTriple::parse("65535.0.1"),
            Some(ReleaseTriple { major: u16::MAX, minor: 0, patch: 1 })
        );
        // Invalid:
        assert_eq!(ReleaseTriple::parse("1.2"), None);
        assert_eq!(ReleaseTriple::parse("1.2.3.4"), None);
        assert_eq!(ReleaseTriple::parse("a.2.3"), None);
        assert_eq!(ReleaseTriple::parse("1.b.3"), None);
        assert_eq!(ReleaseTriple::parse("1.2.c"), None);
        assert_eq!(ReleaseTriple::parse(""), None);
    }

    #[test]
    fn release_zero_minimum_max_testing() {
        assert_eq!(Release::ZERO.value, 0);
        assert_eq!(Release::MINIMUM.triple(), ReleaseTriple { major: 0, minor: 0, patch: 1 });
        assert!(!Release::MINIMUM.testing());
        let dev = Release::from_triple(ReleaseTriple {
            major: Release::DEVELOPMENT_MAJOR,
            minor: 0,
            patch: 0,
        });
        assert!(dev.testing());
        assert_eq!(Release::max(dev, Release::ZERO), dev);
        assert_eq!(Release::max(Release::ZERO, Release::MINIMUM), Release::MINIMUM);
    }
}
