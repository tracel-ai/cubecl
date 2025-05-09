use bytemuck::{Pod, Zeroable};

/// A 8-bit floating point type with 4 exponent bits and 3 mantissa bits.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e4m3(u8);

/// A 8-bit floating point type with 5 exponent bits and 2 mantissa bits.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e5m2(u8);

/// An 8-bit unsigned floating point type with 8 exponent bits and no mantissa bits.
/// Used for scaling factors.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct ue8m0(u8);

impl e4m3 {
    /// Maximum representable value
    pub const MAX: f64 = 240.0;
    /// Minimum representable value
    pub const MIN: f64 = -240.0;
}

impl e5m2 {
    /// Maximum representable value
    pub const MAX: f64 = 57344.0;
    /// Minimum representable value
    pub const MIN: f64 = -57344.0;
}

impl ue8m0 {
    /// Maximum representable value
    pub const MAX: f64 = f64::from_bits(0x47E0000000000000);
    /// Minimum representable value
    pub const MIN: f64 = 0.0;
}
