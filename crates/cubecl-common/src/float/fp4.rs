use bytemuck::{Pod, Zeroable};

/// A 4-bit floating point type with 2 exponent bits and 1 mantissa bit.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, PartialEq, PartialOrd)]
pub struct e2m1(u8);

/// A 4-bit floating point type with 2 exponent bits and 1 mantissa bit. Packed with two elements
/// per value, to allow for conversion to/from bytes. Care must be taken to ensure the shape is
/// adjusted appropriately.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e2m1x2(u8);

impl e2m1 {
    /// Maximum representable value
    pub const MAX: f64 = 3.0;
    /// Minimum representable value
    pub const MIN: f64 = -3.0;
}
