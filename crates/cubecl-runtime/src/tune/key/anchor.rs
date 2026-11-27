use core::sync::atomic::{AtomicI32, Ordering};

use crate::config::autotune::AutotuneLevel;
use crate::config::{CubeClRuntimeConfig, RuntimeConfig};

/// How a key rounds one of its sizes, so that sizes close enough to share a winner share a key:
/// up to the next power of a base the configured [`AutotuneLevel`] scales, within a floor and a
/// ceiling.
///
/// # Guarantees
///
/// - The level is the one the runtime configuration holds when the first anchor of the process is
///   built: every anchor of the process rounds by it, whatever the configuration says later.
/// - [`Full`](AutotuneLevel::Full) rounds nothing: every size is its own key.
///
/// # Examples
///
/// ```
/// use cubecl_runtime::tune::Anchor;
///
/// let size = Anchor::new(Some(1024), Some(16), None).anchor_value(100);
/// assert!((16..=1024).contains(&size));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    max: Option<usize>,
    min: Option<usize>,
    /// The base the level scaled, `None` where the level rounds nothing.
    base: Option<f64>,
}

impl Anchor {
    /// Rounds up to a power of `base`, 2 by default, capped at `max` and floored at `min`.
    pub fn new(max: Option<usize>, min: Option<usize>, base: Option<usize>) -> Self {
        Self {
            max,
            min,
            base: scaled_base(base, configured_level()),
        }
    }

    /// The key size `value` rounds to.
    ///
    /// # Guarantees
    ///
    /// - Where the level rounds nothing, `value` itself, `min` and `max` set aside.
    /// - Otherwise at least `min`, and at most `max` unless `min` is larger. Between them, the
    ///   smallest power of the scaled base, rounded up, that is at least `value`, and `0` for `0`.
    pub fn anchor_value(&self, value: usize) -> usize {
        anchored(value, self.base, self.max, self.min)
    }
}

/// Autotune levels, cached once read:
///
/// '0' => Minimal autotune: scaled anchor of '1.25'.
/// '1' => Medium autotune: normal anchor.
/// '2' => More autotune: scaled anchor of '0.75'.
/// '3' => Autotune everything without anchor.
static AUTOTUNE_LEVEL: AtomicI32 = AtomicI32::new(UNREAD_LEVEL);

/// What [`AUTOTUNE_LEVEL`] holds before the configuration is read.
const UNREAD_LEVEL: i32 = -1;

/// The base below which no power grows fast enough to round anything.
const MIN_BASE: f64 = 1.1;

/// The default base of an anchor.
const DEFAULT_BASE: usize = 2;

/// The level the configuration held when first read.
fn configured_level() -> i32 {
    let cached = AUTOTUNE_LEVEL.load(Ordering::Relaxed);
    if cached != UNREAD_LEVEL {
        return cached;
    }
    let level = level_code(&CubeClRuntimeConfig::get().autotune.level);
    AUTOTUNE_LEVEL.store(level, Ordering::Relaxed);
    level
}

fn level_code(level: &AutotuneLevel) -> i32 {
    match level {
        AutotuneLevel::Minimal => 0,
        AutotuneLevel::Balanced => 1,
        AutotuneLevel::Extensive => 2,
        AutotuneLevel::Full => 3,
    }
}

/// `base`, 2 by default, scaled by the level coded `level`, or `None` where it rounds nothing.
fn scaled_base(base: Option<usize>, level: i32) -> Option<f64> {
    let factor = match level {
        3 => return None,
        2 => 0.75,
        1 => 1.0,
        0 => 1.25,
        _ => unreachable!("the cached autotune level should be one of the four codes"),
    };
    Some(f64::max(
        base.unwrap_or(DEFAULT_BASE) as f64 * factor,
        MIN_BASE,
    ))
}

/// `value` rounded up to a power of `base`, capped at `max` then floored at `min`.
///
/// The exponent is counted rather than taken from a logarithm: `ln(2^29) / ln(2)` is a hair over
/// 29, and its ceiling would round an exact power up to the next one.
fn anchored(value: usize, base: Option<f64>, max: Option<usize>, min: Option<usize>) -> usize {
    let Some(base) = base else {
        return value;
    };
    let power = match value {
        0 => 0,
        _ => base.powi(exponent_reaching(base, value)).ceil() as usize,
    };
    let capped = max.map_or(power, |max| power.min(max));
    min.map_or(capped, |min| capped.max(min))
}

/// The smallest exponent at which a power of `base` reaches `value`.
fn exponent_reaching(base: f64, value: usize) -> i32 {
    let target = value as f64;
    (0..)
        .find(|exponent| base.powi(*exponent) >= target)
        .expect("a power of a base above one should reach every value")
}

#[cfg(test)]
mod tests {
    use super::*;

    const BALANCED: i32 = 1;
    const FULL: i32 = 3;

    fn balanced(value: usize) -> usize {
        anchored(value, scaled_base(None, BALANCED), None, None)
    }

    #[test]
    fn an_exact_power_is_its_own_anchor() {
        assert_eq!(balanced(1 << 29), 1 << 29);
        assert_eq!(balanced(1 << 31), 1 << 31);
        assert_eq!(balanced(1 << 47), 1 << 47);
    }

    #[test]
    fn a_size_rounds_up_to_the_next_power() {
        assert_eq!(balanced(100), 128);
        assert_eq!(balanced(1), 1);
    }

    #[test]
    fn a_zero_size_anchors_to_the_floor() {
        assert_eq!(balanced(0), 0);
        assert_eq!(anchored(0, scaled_base(None, BALANCED), None, Some(16)), 16);
    }

    #[test]
    fn the_full_level_rounds_nothing() {
        assert_eq!(
            anchored(100, scaled_base(None, FULL), Some(16), Some(8)),
            100
        );
    }

    #[test]
    fn the_ceiling_caps_and_the_floor_wins_over_it() {
        let base = scaled_base(None, BALANCED);
        assert_eq!(anchored(100, base, Some(64), None), 64);
        assert_eq!(anchored(3, base, None, Some(16)), 16);
        assert_eq!(anchored(100, base, Some(8), Some(16)), 16);
    }

    #[test]
    fn a_scaled_base_never_drops_under_the_minimum() {
        assert_eq!(scaled_base(Some(1), 2), Some(MIN_BASE));
    }
}
