use cubecl_core::prelude::*;

/// Useful for functions that may change their implementation depending on quantization.
/// This trait can be used to keep a simple signature and then dispatch on depending on `QUANTIZED`.
///
/// Currently, this is implemented with `QUANTIZED == false` for all types that implement `Numeric`.
/// To add a new quantized type, one should create a static type. See [Q8] for example.
///
/// # Examples
///
/// ```ignored
///
/// pub fn double_if_quantized<N: MaybeQuantized>(x: N::Numeric) -> N::Numeric {
///     if N::QUANTIZED {
///         x * N::Numeric::from_int(2)
///     } else {
///         x
///     }
/// }
///
/// // At call site (similar for cube functions).
///
/// let y = double_if_quantized::<u8>(3);
/// assert_eq!(y, 3);
///
/// let z = double_if_quantized::<Q8>(3);
/// assert_eq!(z, 6);
/// ```
pub trait MaybeQuantized {
    type Numeric: Numeric;
    const QUANTIZED: bool;
}

impl<N: Numeric> MaybeQuantized for N {
    type Numeric = N;
    const QUANTIZED: bool = false;
}

/// Represent the quantization of a f32 into an i8 using the symmetric scheme.
pub struct SymQ8;

impl MaybeQuantized for SymQ8 {
    type Numeric = i8;
    const QUANTIZED: bool = true;
}

impl SymQ8 {
    pub fn quantize(val: f32, scaling: f32) -> i8 {
        let min = -scaling * (i8::MIN as f32);
        let max = min + 255.0 * scaling;
        if val < min {
            i8::MIN
        } else if val > max {
            i8::MAX
        } else {
            ((val - min) / scaling).round() as i8
        }
    }

    pub fn dequantize(val: i8, scaling: f32) -> f32 {
        val as f32 * scaling
    }
}
