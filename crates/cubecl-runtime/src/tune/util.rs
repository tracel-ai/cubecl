/// Anchor a number to a power of the provided base.
///
/// Useful when creating autotune keys.
pub fn anchor(x: usize, max: Option<usize>, min: Option<usize>, base: Option<usize>) -> usize {
    let base = base.unwrap_or(2);
    let exp = usize::ilog(x, base);
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
