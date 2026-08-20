use cubecl_common::{e4m3, e5m2};
use enumset::EnumSetType;
use pliron::{context::Context, r#type::TypeHandle};

use crate::types::scalar::{Float8E4M3Type, Float8E5M2Type};

/// Bit mask of the seven magnitude bits of an fp8 code.
const FP8_MAGNITUDE_MASK: u32 = 0x7F;

/// The two IEEE-style fp8 formats, described by the field constants their host codecs publish.
#[derive(Debug, Hash, EnumSetType)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

impl Fp8Format {
    pub const fn exponent_bits(self) -> u32 {
        7 - self.mantissa_bits()
    }

    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => e4m3::MANTISSA_DIGITS - 1,
            Fp8Format::E5M2 => e5m2::MANTISSA_DIGITS - 1,
        }
    }

    pub const fn bias(self) -> u32 {
        let min_exp = match self {
            Fp8Format::E4M3 => e4m3::MIN_EXP,
            Fp8Format::E5M2 => e5m2::MIN_EXP,
        };
        (2 - min_exp) as u32
    }

    pub const fn max_value(self) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::MAX.to_f32(),
            Fp8Format::E5M2 => e5m2::MAX.to_f32(),
        }
    }

    pub const fn max_code(self) -> u32 {
        match self {
            Fp8Format::E4M3 => e4m3::MAX.to_bits() as u32,
            Fp8Format::E5M2 => e5m2::MAX.to_bits() as u32,
        }
    }

    /// Both formats have several NaN encodings and the payload is not carried across: the codecs
    /// truncate the source mantissa, the hardware paths do as they please. This is the code for a
    /// canonical NaN, which is what the codec returns for `f32::NAN`. Reading `NAN` instead would
    /// give e5m2 a payload no conversion produces.
    pub const fn nan_code(self) -> u32 {
        let nan = match self {
            Fp8Format::E4M3 => e4m3::from_f32(f32::NAN).to_bits(),
            Fp8Format::E5M2 => e5m2::from_f32(f32::NAN).to_bits(),
        };
        nan as u32 & FP8_MAGNITUDE_MASK
    }

    pub const fn has_infinity(self) -> bool {
        self.decode(self.max_code() + 1).is_infinite()
    }

    pub const fn min_normal(self) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::MIN_POSITIVE.to_f32(),
            Fp8Format::E5M2 => e5m2::MIN_POSITIVE.to_f32(),
        }
    }

    pub const fn subnormal_step(self) -> f32 {
        self.decode(1)
    }

    const fn decode(self, code: u32) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::from_bits(code as u8).to_f32(),
            Fp8Format::E5M2 => e5m2::from_bits(code as u8).to_f32(),
        }
    }

    pub fn of_type(ctx: &Context, ty: TypeHandle) -> Option<Self> {
        let ty = ty.deref(ctx);
        if ty.is::<Float8E4M3Type>() {
            Some(Fp8Format::E4M3)
        } else if ty.is::<Float8E5M2Type>() {
            Some(Fp8Format::E5M2)
        } else {
            None
        }
    }
}
