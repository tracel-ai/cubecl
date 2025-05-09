use bytemuck::{Pod, Zeroable};

/// A 6-bit floating point type with 2 exponent bits and 3 mantissa bits.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e2m3(u8);

/// A 6-bit floating point type with 3 exponent bits and 2 mantissa bits.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e3m2(u8);

impl e2m3 {
    /// Maximum representable value
    pub const MAX: f64 = 3.75;
    /// Minimum representable value
    pub const MIN: f64 = -3.75;
}

impl e3m2 {
    /// Maximum representable value
    pub const MAX: f64 = 14.0;
    /// Minimum representable value
    pub const MIN: f64 = -14.0;
}
