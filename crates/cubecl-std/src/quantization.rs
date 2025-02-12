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

/// Represent the quantization of a f32 into a u8.
pub struct Q8;

impl MaybeQuantized for Q8 {
    type Numeric = u8;
    const QUANTIZED: bool = true;
}

impl Q8 {
    pub fn quantize(val: f32, scaling: f32, zero_offset: u8) -> u8 {
        let min = -scaling * (zero_offset as f32);
        let max = min + 255.0 * scaling;
        if val < min {
            0
        } else if val > max {
            255
        } else {
            ((val - min) / scaling).round() as u8
        }
    }

    pub fn dequantize(val: u8, scaling: f32, zero_offset: u8) -> f32 {
        (val as i16 - zero_offset as i16) as f32 * scaling
    }
}
