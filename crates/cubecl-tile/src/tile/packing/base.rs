//! How an operand's values sit in memory.

use cubecl::client::Client;
use cubecl::e2m1x2;
use cubecl::ir::features::TypeUsage;
use cubecl::ir::types::Fp8Format;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::post_processing::fp4::E2M1_F16_LIFT;
use cubecl::prelude::Scalar;
use cubecl::quant::scheme::QuantValue;
use cubecl::quant::scheme::ScaleDtype;

/// How an operand's values sit in memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Packing {
    /// Served as stored.
    Plain,
    /// Several `field`-wide values per stored `u32`, unpacked at the read.
    Packed {
        /// The slot one value occupies.
        field: Field,
    },
}

/// The slot one packed value occupies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Field {
    Quant(QuantValue),
    Fp8(Fp8Format),
    Float(FloatKind),
    /// An index `bits` wide into a table, its codebook
    /// ([`TileArgLaunch::codebook`](crate::TileArgLaunch::codebook)), read as raw bits.
    Index {
        bits: usize,
    },
    /// An `e2m1` code on a device that converts `e2m1x2` itself: a word is read as four of them
    /// and cast, the decode left to the compiler. What [`read_on`](Field::read_on) makes of
    /// `Quant(E2M1)` there.
    ConvertedE2M1,
    /// An `e2m1` code on a device that emulates `e2m1x2`'s conversion: a word is decoded as `f16`
    /// pairs from its bits, lifted to its values, or, where `lifted` is false, each placed
    /// [`E2M1_F16_LIFT`] short of its value for a reader whose factor carries the lift (a tile
    /// read placed). What [`read_on`](Field::read_on) makes of
    /// `Quant(E2M1)` there, lifted.
    PlacedE2M1 {
        lifted: bool,
    },
}

impl From<QuantValue> for Field {
    fn from(value: QuantValue) -> Self {
        Field::Quant(value)
    }
}

impl From<Fp8Format> for Field {
    fn from(format: Fp8Format) -> Self {
        Field::Fp8(format)
    }
}

/// An 8-bit float is a [`Fp8`](Field::Fp8) field, a wider one a [`Float`](Field::Float) field.
impl From<FloatKind> for Field {
    fn from(kind: FloatKind) -> Self {
        Field::of_float(kind)
    }
}

impl Field {
    /// The slot's width in bits.
    pub fn size_bits(self) -> usize {
        match self {
            Field::Quant(value) => value.size_bits(),
            Field::Fp8(_) => 8,
            Field::Float(kind) => Field::float_bits(kind),
            Field::Index { bits } => bits,
            Field::ConvertedE2M1 | Field::PlacedE2M1 { .. } => QuantValue::E2M1.size_bits(),
        }
    }

    /// Values one `u32` holds.
    pub fn per_word(self) -> usize {
        u32::BITS as usize / self.size_bits()
    }
}

impl Field {
    /// This field as the device `client` reads it: an `e2m1` code decoded from its bits where the
    /// device emulates converting `e2m1x2` anyway, so a reader's factor can carry the decode's
    /// lift; converted by the device where it converts `e2m1x2` itself; every other field as it
    /// is. A device's own conversion is the one its compiler lowers best, so where it exists a
    /// software decode is only ever slower.
    pub fn read_on(self, client: &Client) -> Field {
        let e2m1 = ElemType::Float(FloatKind::E2M1x2);
        let emulated = client.features().conversion_is_emulated(e2m1);
        let converts = e2m1x2::supported_uses(client).contains(TypeUsage::Conversion);
        match self {
            Field::Quant(QuantValue::E2M1) if emulated => Field::PlacedE2M1 { lifted: true },
            Field::Quant(QuantValue::E2M1) if converts => Field::ConvertedE2M1,
            other => other,
        }
    }

    /// This field read placed ([`Tile::placed`](crate::Tile::placed)): an `e2m1` code decoded
    /// from its bits, its lift left to the reader's factor; every other field as it is.
    pub(crate) fn placed(self) -> Field {
        match self {
            Field::PlacedE2M1 { .. } => Field::PlacedE2M1 { lifted: false },
            other => other,
        }
    }

    /// What a value read this way is short of: [`E2M1_F16_LIFT`] for an `e2m1` code placed, one
    /// for every value read as itself.
    pub(crate) fn lift(self) -> f64 {
        match self {
            Field::PlacedE2M1 { lifted: false } => E2M1_F16_LIFT,
            _ => 1.0,
        }
    }

    /// The field a float of `kind` occupies.
    pub fn of_float(kind: FloatKind) -> Field {
        match kind {
            FloatKind::E4M3 => Field::Fp8(Fp8Format::E4M3),
            FloatKind::E5M2 => Field::Fp8(Fp8Format::E5M2),
            FloatKind::UE8M0 => Field::Fp8(Fp8Format::UE8M0),
            other => Field::Float(other),
        }
    }

    /// The field a scale of `dtype` occupies in its stored word.
    pub fn of_scale(dtype: ScaleDtype) -> Field {
        match ElemType::from_scale_dtype(dtype) {
            ElemType::Float(kind) => Field::of_float(kind),
            other => panic!("Field: a scale is stored as a float, and {dtype:?} is {other:?}"),
        }
    }

    /// The bits a whole float occupies.
    pub(crate) fn float_bits(kind: FloatKind) -> usize {
        match kind {
            FloatKind::F32 => 32,
            FloatKind::F16 | FloatKind::BF16 => 16,
            other => panic!(
                "Field::Float: a stored float is 16 or 32 bits wide, got {other:?}; an 8-bit \
                 code is Field::Fp8"
            ),
        }
    }

    /// How this field reads back.
    pub(crate) fn decode(self) -> FieldDecode {
        match self {
            Field::Quant(
                QuantValue::Q8F
                | QuantValue::Q8S
                | QuantValue::Q4F
                | QuantValue::Q4S
                | QuantValue::Q2F
                | QuantValue::Q2S,
            ) => FieldDecode::SignExtended,
            Field::Quant(QuantValue::E2M1) => FieldDecode::Reinterpreted,
            Field::Quant(QuantValue::E4M3) => FieldDecode::Byte(Fp8Format::E4M3),
            Field::Quant(QuantValue::E5M2) => FieldDecode::Byte(Fp8Format::E5M2),
            Field::Fp8(format) => FieldDecode::Byte(format),
            Field::Float(kind) => FieldDecode::Bits(kind),
            Field::Index { .. } => FieldDecode::Unsigned,
            Field::ConvertedE2M1 => FieldDecode::Converted,
            Field::PlacedE2M1 { lifted } => FieldDecode::Placed { lifted },
        }
    }
}

/// How a stored field reads back ([`Field::decode`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FieldDecode {
    /// An integer slot: the top bit is its sign, so the value sign-extends out of its bits.
    SignExtended,
    /// An index slot: its bits, unsigned.
    Unsigned,
    /// A 4-bit float code, read back by reinterpreting the byte two of them share.
    Reinterpreted,
    /// A 4-bit float code the device converts: a word cast as four `e2m1x2`.
    Converted,
    /// A 4-bit float code decoded from its bits as `f16` pairs, lifted to its value or placed
    /// [`E2M1_F16_LIFT`] short of it.
    Placed { lifted: bool },
    /// An 8-bit float code, read back through the format's own decoder.
    Byte(Fp8Format),
    /// A whole float, read back by reinterpreting the bits of its slot.
    Bits(FloatKind),
}

impl Packing {
    /// This packing read placed ([`Field::placed`]).
    pub(crate) fn placed(self) -> Packing {
        match self {
            Packing::Packed { field } => Packing::Packed {
                field: field.placed(),
            },
            Packing::Plain => Packing::Plain,
        }
    }

    /// What its values are short of ([`Field::lift`]).
    pub(crate) fn lift(self) -> f64 {
        match self {
            Packing::Packed { field } => field.lift(),
            Packing::Plain => 1.0,
        }
    }

    /// This packing as the device `client` reads it ([`Field::read_on`]).
    pub(crate) fn read_on(self, client: &Client) -> Packing {
        match self {
            Packing::Packed { field } => Packing::Packed {
                field: field.read_on(client),
            },
            Packing::Plain => Packing::Plain,
        }
    }
}

impl Packing {
    /// Values per stored element: one, unless a `u32` holds several fields.
    fn factor(&self) -> usize {
        match self {
            Packing::Plain => 1,
            Packing::Packed { field } => field.per_word(),
        }
    }

    /// The physical line a `served`-wide logical line occupies.
    pub(crate) fn physical(&self, served: usize) -> usize {
        served / self.factor()
    }

    /// The width a binding `bound` wide serves.
    pub(crate) fn served(&self, bound: usize) -> usize {
        bound * self.factor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packing_narrows_the_line_it_stores() {
        assert_eq!(Packing::Plain.physical(16), 16);
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q4S.into()
            }
            .physical(16),
            2
        );
    }

    #[test]
    fn a_field_states_how_many_fit_in_a_word() {
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q4S.into()
            }
            .factor(),
            8
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q8S.into()
            }
            .factor(),
            4
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q2S.into()
            }
            .factor(),
            16
        );
        assert_eq!(
            Packing::Packed {
                field: Fp8Format::UE8M0.into()
            }
            .factor(),
            4
        );
    }

    #[test]
    fn a_float_element_names_its_own_field() {
        assert_eq!(
            Field::of_float(FloatKind::UE8M0),
            Field::Fp8(Fp8Format::UE8M0)
        );
        assert_eq!(
            Field::of_float(FloatKind::E4M3),
            Field::Fp8(Fp8Format::E4M3)
        );
        assert_eq!(
            Field::of_float(FloatKind::F16),
            Field::Float(FloatKind::F16)
        );
    }

    #[test]
    fn a_stored_float_is_a_field_of_its_own_width() {
        assert_eq!(Field::Float(FloatKind::F32).per_word(), 1);
        assert_eq!(Field::Float(FloatKind::F16).per_word(), 2);
        assert_eq!(Field::Float(FloatKind::BF16).size_bits(), 16);
        assert_eq!(
            Field::from(FloatKind::F32).decode(),
            FieldDecode::Bits(FloatKind::F32)
        );
    }

    #[test]
    fn a_byte_field_decodes_through_its_format() {
        assert_eq!(
            Field::from(QuantValue::E4M3).decode(),
            FieldDecode::Byte(Fp8Format::E4M3)
        );
        assert_eq!(
            Field::from(Fp8Format::UE8M0).decode(),
            FieldDecode::Byte(Fp8Format::UE8M0)
        );
        assert_eq!(
            Field::from(QuantValue::E2M1).decode(),
            FieldDecode::Reinterpreted
        );
        assert_eq!(Field::ConvertedE2M1.decode(), FieldDecode::Converted);
        assert_eq!(Field::ConvertedE2M1.per_word(), 8);
    }
}
