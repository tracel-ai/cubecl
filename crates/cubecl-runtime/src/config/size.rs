use alloc::format;
use alloc::string::String;
use core::fmt;

/// A byte count that deserializes from either a raw integer (bytes) or a
/// human-readable string such as `"8KiB"`, `"512MB"`, `"1.5GiB"`.
///
/// All suffixes are binary (powers of 1024): `K`/`KB`/`KiB` = 1024,
/// `M`/`MB`/`MiB` = 1024², `G`/`GB`/`GiB` = 1024³, `T`/`TB`/`TiB` = 1024⁴,
/// case-insensitive, with optional whitespace before the unit. A fractional
/// number is allowed with a unit (`"1.5GiB"`); the result is rounded down to a
/// whole byte. Serializes as a plain integer for lossless round-trips.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemorySize(pub u64);

impl MemorySize {
    /// The size in bytes.
    pub const fn bytes(self) -> u64 {
        self.0
    }
}

impl core::str::FromStr for MemorySize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let split = s.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '_');
        let (num, unit) = match split {
            Some(i) => s.split_at(i),
            None => (s, ""),
        };
        let unit = unit.trim();
        let multiplier: u64 = match unit.to_ascii_lowercase().as_str() {
            "" | "b" => 1,
            "k" | "kb" | "kib" => 1 << 10,
            "m" | "mb" | "mib" => 1 << 20,
            "g" | "gb" | "gib" => 1 << 30,
            "t" | "tb" | "tib" => 1 << 40,
            other => return Err(format!("unknown size unit `{other}` in `{s}`")),
        };

        // Integer fast path preserves full u64 precision; the f64 path is only
        // for fractional values.
        let num_clean = num.replace('_', "");
        let bytes = if let Ok(value) = num_clean.parse::<u64>() {
            value
                .checked_mul(multiplier)
                .ok_or_else(|| format!("size `{s}` overflows u64"))?
        } else {
            let value: f64 = num_clean
                .parse()
                .map_err(|_| format!("invalid number `{num}` in size `{s}`"))?;
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("invalid size `{s}`"));
            }
            let bytes = value * multiplier as f64;
            if bytes > u64::MAX as f64 {
                return Err(format!("size `{s}` overflows u64"));
            }
            bytes as u64
        };

        Ok(MemorySize(bytes))
    }
}

impl fmt::Display for MemorySize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl serde::Serialize for MemorySize {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for MemorySize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = MemorySize;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a byte count as an integer or a string like \"8KiB\"")
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<MemorySize, E> {
                Ok(MemorySize(value))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<MemorySize, E> {
                u64::try_from(value)
                    .map(MemorySize)
                    .map_err(|_| E::custom("byte size cannot be negative"))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<MemorySize, E> {
                value.parse().map_err(E::custom)
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<u64, String> {
        s.parse::<MemorySize>().map(MemorySize::bytes)
    }

    #[test]
    fn parses_integers_and_units() {
        assert_eq!(parse("1024"), Ok(1024));
        assert_eq!(parse("0"), Ok(0));
        assert_eq!(parse("8KiB"), Ok(8 * 1024));
        assert_eq!(parse("20GiB"), Ok(20 * 1024 * 1024 * 1024));
        assert_eq!(parse("512 mb"), Ok(512 * 1024 * 1024));
        assert_eq!(parse("1.5GiB"), Ok(1024 * 1024 * 1024 * 3 / 2));
        assert_eq!(parse("64k"), Ok(64 * 1024));
        assert_eq!(parse("2TiB"), Ok(2u64 << 40));
        assert_eq!(parse("1_048_576"), Ok(1 << 20));
    }

    #[test]
    fn rejects_invalid_sizes() {
        assert!(parse("abc").is_err());
        assert!(parse("-5").is_err());
        assert!(parse("5XB").is_err());
        assert!(parse("99999999999999999999TiB").is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn serde_accepts_integers_and_strings() {
        let from_int: MemorySize = serde_json::from_str("8192").unwrap();
        assert_eq!(from_int.bytes(), 8192);

        let from_str: MemorySize = serde_json::from_str("\"8KiB\"").unwrap();
        assert_eq!(from_str.bytes(), 8192);

        assert!(serde_json::from_str::<MemorySize>("-1").is_err());

        // Serializes as a plain integer.
        assert_eq!(serde_json::to_string(&MemorySize(8192)).unwrap(), "8192");
    }
}
