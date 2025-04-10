use core::sync::atomic::{AtomicI32, Ordering};

static AUTOTUNE_LEVEL: AtomicI32 = AtomicI32::new(-1);

/// Anchor a number to a power of the provided base.
///
/// Useful when creating autotune keys.
pub fn anchor(x: usize, max: Option<usize>, min: Option<usize>, base: Option<usize>) -> usize {
    let autotune_level = AUTOTUNE_LEVEL.load(Ordering::Relaxed);
    let level = if autotune_level == -1 {
        #[cfg(feature = "std")]
        {
            let level: u32 = std::env::var("CUBECL_AUTOTUNE_LEVEL")
                .map(|value| {
                    value
                        .parse()
                        .expect("'CUBECL_AUTOTUNE_LEVEL' should be an integer.")
                })
                .unwrap_or(1);
            AUTOTUNE_LEVEL.store(level as i32, Ordering::Relaxed);
            level
        }
        #[cfg(not(feature = "std"))]
        1
    } else {
        autotune_level as u32
    };

    if level == 1 {
        return x;
    }

    let base = base.unwrap_or(2);
    let exp = (x as f64).log(base as f64).ceil() as u32;
    let power = base.pow(exp);

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
