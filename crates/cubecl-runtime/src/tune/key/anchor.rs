use core::sync::atomic::{AtomicI32, Ordering};

use crate::config::autotune::AutotuneLevel;
use crate::config::{CubeClRuntimeConfig, RuntimeConfig};

/// How a key rounds one of its sizes, so that sizes close enough to share a decision share a
/// key: up to the next power of a base scaled by the configured [`AutotuneLevel`], within a floor
/// and a ceiling.
///
/// # Examples
///
/// ```
/// use cubecl_runtime::tune::Anchor;
///
/// let size = Anchor { max: Some(1024), min: Some(16), ..Anchor::default() }.round(100);
/// assert!((16..=1024).contains(&size));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Anchor {
    /// The ceiling, if any.
    pub max: Option<usize>,
    /// The floor, if any.
    pub min: Option<usize>,
    /// The base before the level scales it, 2 by default.
    pub base: Option<usize>,
}

impl Anchor {
    /// The key size `size` rounds to.
    ///
    /// # Guarantees
    ///
    /// - The level is the one the runtime configuration holds when the process first rounds a
    ///   size: every size of the process rounds by it, whatever the configuration says later.
    /// - [`Full`](AutotuneLevel::Full) rounds nothing: every size is its own key, `min` and `max`
    ///   set aside.
    /// - Otherwise at least `min`, and at most `max` unless `min` is larger. Between them, the
    ///   smallest power of the scaled base, rounded up, that is at least `size`, and `0` for `0`.
    pub fn round(&self, size: usize) -> usize {
        anchored(
            size,
            scaled_base(self.base, configured_level()),
            self.max,
            self.min,
        )
    }
}

/// The code of the autotune level the configuration held when first read, [`UNREAD_LEVEL`]
/// before: one of [`level_code`]'s.
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

/// The base the level coded `level` scales, or `None` where it rounds nothing.
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

/// [`Anchor::round`], over a base already scaled.
fn anchored(size: usize, base: Option<f64>, max: Option<usize>, min: Option<usize>) -> usize {
    let Some(base) = base else {
        return size;
    };
    let power = match size {
        0 => 0,
        _ => power_reaching(base, size),
    };
    let capped = max.map_or(power, |max| power.min(max));
    min.map_or(capped, |min| capped.max(min))
}

/// The smallest power of `base`, rounded up, that is at least `size`, saturating at
/// [`usize::MAX`].
///
/// The exponent is counted rather than taken from a logarithm: `ln(2^29) / ln(2)` is a hair over
/// 29, and its ceiling would round an exact power up to the next one. Each power is rounded up
/// and compared as an integer, since `size` as a float rounds down above 2^53.
fn power_reaching(base: f64, size: usize) -> usize {
    (0..)
        .map(|exponent| rounded_up(base.powi(exponent)))
        .find(|power| *power >= size)
        .expect("a power of a base above one should reach every size")
}

/// `power` rounded up to an integer, saturating at [`usize::MAX`].
fn rounded_up(power: f64) -> usize {
    // A float cast saturates: every power past the last `usize` is `usize::MAX`.
    power.ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(level: AutotuneLevel) -> i32 {
        level_code(&level)
    }

    fn balanced(size: usize) -> usize {
        anchored(
            size,
            scaled_base(None, code(AutotuneLevel::Balanced)),
            None,
            None,
        )
    }

    #[test]
    fn an_exact_power_is_its_own_anchor() {
        assert_eq!(balanced(1 << 29), 1 << 29);
    }

    #[test]
    fn an_exact_power_past_2_to_the_31_is_its_own_anchor() {
        assert_eq!(balanced(1 << 47), 1 << 47);
    }

    #[test]
    fn a_size_rounds_up_to_the_next_power() {
        assert_eq!(balanced(100), 128);
    }

    #[test]
    fn one_is_its_own_anchor() {
        assert_eq!(balanced(1), 1);
    }

    #[test]
    fn a_size_past_2_to_the_53_rounds_up_to_the_next_power() {
        assert_eq!(balanced((1 << 53) + 1), 1 << 54);
    }

    #[test]
    fn a_size_past_the_last_power_anchors_to_the_largest_size() {
        assert_eq!(balanced(usize::MAX), usize::MAX);
    }

    #[test]
    fn a_zero_size_anchors_to_zero() {
        assert_eq!(balanced(0), 0);
    }

    #[test]
    fn a_zero_size_anchors_to_the_floor() {
        let base = scaled_base(None, code(AutotuneLevel::Balanced));

        assert_eq!(anchored(0, base, None, Some(16)), 16);
    }

    #[test]
    fn the_full_level_rounds_nothing() {
        let base = scaled_base(None, code(AutotuneLevel::Full));

        assert_eq!(anchored(100, base, Some(16), Some(8)), 100);
    }

    #[test]
    fn the_ceiling_caps() {
        let base = scaled_base(None, code(AutotuneLevel::Balanced));

        assert_eq!(anchored(100, base, Some(64), None), 64);
    }

    #[test]
    fn the_floor_raises() {
        let base = scaled_base(None, code(AutotuneLevel::Balanced));

        assert_eq!(anchored(3, base, None, Some(16)), 16);
    }

    #[test]
    fn the_floor_wins_over_the_ceiling() {
        let base = scaled_base(None, code(AutotuneLevel::Balanced));

        assert_eq!(anchored(100, base, Some(8), Some(16)), 16);
    }

    #[test]
    fn a_scaled_base_never_drops_under_the_minimum() {
        assert_eq!(
            scaled_base(Some(1), code(AutotuneLevel::Extensive)),
            Some(MIN_BASE)
        );
    }
}
