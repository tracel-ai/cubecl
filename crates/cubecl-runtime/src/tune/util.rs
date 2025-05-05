use core::sync::atomic::{AtomicI32, Ordering};

use crate::config::GlobalConfig;

/// Autotune levels:
///
/// '0' => Minimal autotune: scaled anchor of '1.25'.
/// '1' => Medium autotune: normal anchor.
/// '2' => More autotune: scaled anchor of '0.75'.
/// '3' => Autotune everything without anchor.
static AUTOTUNE_LEVEL: AtomicI32 = AtomicI32::new(-1);

/// Anchor a number to a power of the provided base.
///
/// Useful when creating autotune keys.
pub fn anchor(x: usize, max: Option<usize>, min: Option<usize>, base: Option<usize>) -> usize {
    let autotune_level = load_autotune_level();
    let factor = match autotune_level {
        3 => return x, // Autotune everything, there is no anchor.
        2 => 0.75,
        1 => 1.0,
        0 => 1.25,
        _ => panic!("Invalid autotune level {autotune_level:?}"),
    };

    let base = base.unwrap_or(2) as f64 * factor;
    let base = f64::max(base, 1.1); // Minimum base.
    let exp = (x as f64).log(base).ceil();
    let power = base.powf(exp).ceil() as usize;

    let result = if let Some(max) = max {
        core::cmp::min(power, max)
    } else {
        power
    };

    if let Some(min) = min {
        core::cmp::max(result, min)
    } else {
        result
    }
}

fn load_autotune_level() -> u32 {
    let autotune_level = AUTOTUNE_LEVEL.load(Ordering::Relaxed);
    if autotune_level == -1 {
        let config = GlobalConfig::get();
        let level = match config.autotune.level {
            crate::config::autotune::AutotuneLevel::Minimal => 0,
            crate::config::autotune::AutotuneLevel::Balanced => 1,
            crate::config::autotune::AutotuneLevel::Extensive => 2,
            crate::config::autotune::AutotuneLevel::Full => 3,
        };
        AUTOTUNE_LEVEL.store(level, Ordering::Relaxed);
        level as u32
    } else {
        autotune_level as u32
    }
}
