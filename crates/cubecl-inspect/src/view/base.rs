use std::fmt;
use std::time::Duration;

/// A report, rendered as text by its `Display` implementation: plain, or
/// painted for a terminal in its alternate form, `{:#}`.
pub struct Text<'a, R>(pub &'a R);

/// A byte count at the scale a person reads it.
pub struct ByteSize(pub u64);

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];

        if self.0 < 1024 {
            return write!(f, "{} B", self.0);
        }
        let mut value = self.0 as f64 / 1024.0;
        let mut unit = UNITS[0];
        for next in &UNITS[1..] {
            if value < 1024.0 {
                break;
            }
            value /= 1024.0;
            unit = next;
        }
        write!(f, "{value:.1} {unit}")
    }
}

/// A kernel-scale duration: microseconds, with milliseconds past ten of them.
pub struct Micros(pub Duration);

impl fmt::Display for Micros {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let micros = self.0.as_secs_f64() * 1e6;
        if micros >= 10_000.0 {
            write!(f, "{:.2} ms", micros / 1000.0)
        } else {
            write!(f, "{micros:.1} µs")
        }
    }
}

/// A session-scale duration: milliseconds, then seconds, then minutes.
pub struct Wall(pub Duration);

impl fmt::Display for Wall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs_f64();
        if secs < 1.0 {
            write!(f, "{:.0} ms", secs * 1000.0)
        } else if secs < 120.0 {
            write!(f, "{secs:.1} s")
        } else {
            write!(f, "{:.1} min", secs / 60.0)
        }
    }
}

/// A value that may be missing, `-` when it is.
pub struct Maybe<T>(pub Option<T>);

impl<T: fmt::Display> fmt::Display for Maybe<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(value) => value.fmt(f),
            None => f.write_str("-"),
        }
    }
}

/// A ratio such as a margin or a slowdown, `-` when there is none.
pub struct Ratio(pub Option<f64>);

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(ratio) => write!(f, "{ratio:.2}×"),
            None => f.write_str("-"),
        }
    }
}

/// A fraction as a percentage, `-` when there is none: to the formatter's
/// precision, whole percents by default.
pub struct Percent(pub Option<f64>);

impl Percent {
    /// `part` over `whole`; none when the whole is empty.
    pub fn new(part: u64, whole: u64) -> Self {
        Self((whole > 0).then(|| part as f64 / whole as f64))
    }
}

impl fmt::Display for Percent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(fraction) => {
                let digits = f.precision().unwrap_or(0);
                write!(f, "{:.digits$}%", fraction * 100.0)
            }
            None => f.write_str("-"),
        }
    }
}

/// ` via <fingerprint>` when the table names the adapter and API the device
/// resolved to, nothing when it predates them.
pub struct Via<'a>(pub Option<&'a str>);

impl fmt::Display for Via<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(fingerprint) => write!(f, " via {fingerprint}"),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_scale_to_the_largest_unit_under_1024() {
        assert_eq!(ByteSize(512).to_string(), "512 B");
        assert_eq!(ByteSize(1536).to_string(), "1.5 KiB");
        assert_eq!(ByteSize(5 * 1024 * 1024).to_string(), "5.0 MiB");
    }

    #[test]
    fn a_percent_takes_the_formatter_s_precision() {
        assert_eq!(Percent(Some(0.25)).to_string(), "25%");
        assert_eq!(format!("{:.1}", Percent::new(1, 3)), "33.3%");
        assert_eq!(Percent::new(1, 0).to_string(), "-");
    }

    #[test]
    fn walls_scale_from_milliseconds_to_minutes() {
        assert_eq!(Wall(Duration::from_millis(41)).to_string(), "41 ms");
        assert_eq!(Wall(Duration::from_millis(38_400)).to_string(), "38.4 s");
        assert_eq!(Wall(Duration::from_secs(300)).to_string(), "5.0 min");
    }

    #[test]
    fn durations_switch_to_milliseconds_past_ten() {
        assert_eq!(Micros(Duration::from_nanos(33_703)).to_string(), "33.7 µs");
        assert_eq!(
            Micros(Duration::from_micros(12_500)).to_string(),
            "12.50 ms"
        );
    }
}
