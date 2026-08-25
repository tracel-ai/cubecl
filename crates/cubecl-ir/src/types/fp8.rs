use cubecl_common::{e4m3, e5m2, ue8m0};
use enumset::EnumSetType;
use pliron::{context::Context, r#type::TypeHandle};

use crate::types::scalar::{Float8E4M3Type, Float8E5M2Type, Float8E8M0Type};

/// Bit mask of the seven magnitude bits of an fp8 code.
const FP8_MAGNITUDE_MASK: u32 = 0x7F;

/// The 8-bit float formats, described by the field constants their host codecs publish.
///
/// [`Fp8Format::UE8M0`] is the odd one: it carries no sign and no mantissa, so it is a bare
/// exponent rather than an IEEE-style minifloat. It belongs here anyway because the question this
/// enum answers — which 8-bit formats does the backend convert natively, and which need the
/// software path — is the same question for it, and it is the scale type MXFP4 is defined on.
/// The generic field arithmetic below does not serve it; the codec branches on it instead.
#[derive(Debug, Hash, EnumSetType)]
pub enum Fp8Format {
    E4M3,
    E5M2,
    UE8M0,
}

impl Fp8Format {
    pub const fn exponent_bits(self) -> u32 {
        match self {
            // Eight bits and no sign, so the usual `7 - mantissa` does not describe it.
            Fp8Format::UE8M0 => 8,
            _ => 7 - self.mantissa_bits(),
        }
    }

    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => e4m3::MANTISSA_DIGITS - 1,
            Fp8Format::E5M2 => e5m2::MANTISSA_DIGITS - 1,
            Fp8Format::UE8M0 => 0,
        }
    }

    pub const fn bias(self) -> u32 {
        let min_exp = match self {
            Fp8Format::E4M3 => e4m3::MIN_EXP,
            Fp8Format::E5M2 => e5m2::MIN_EXP,
            // Code 127 is 2^0, so the bias is 127. Spelled out because `ue8m0` publishes no
            // `MIN_EXP`: it has no subnormals for one to describe.
            Fp8Format::UE8M0 => return 127,
        };
        (2 - min_exp) as u32
    }

    pub const fn max_value(self) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::MAX.to_f32(),
            Fp8Format::E5M2 => e5m2::MAX.to_f32(),
            // 2^127. Written as a bit pattern because `to_f32` on `ue8m0` is not `const`.
            Fp8Format::UE8M0 => f32::from_bits(0x7F00_0000),
        }
    }

    pub const fn max_code(self) -> u32 {
        match self {
            Fp8Format::E4M3 => e4m3::MAX.to_bits() as u32,
            Fp8Format::E5M2 => e5m2::MAX.to_bits() as u32,
            Fp8Format::UE8M0 => ue8m0::MAX.to_bits() as u32,
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
            // The all-ones code, one past the maximum. Unsigned, so no magnitude mask applies.
            Fp8Format::UE8M0 => return 0xFF,
        };
        nan as u32 & FP8_MAGNITUDE_MASK
    }

    pub const fn has_infinity(self) -> bool {
        match self {
            // Its one code past the maximum is the NaN, so there is no infinity to reach.
            Fp8Format::UE8M0 => false,
            _ => self.decode(self.max_code() + 1).is_infinite(),
        }
    }

    pub const fn min_normal(self) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::MIN_POSITIVE.to_f32(),
            Fp8Format::E5M2 => e5m2::MIN_POSITIVE.to_f32(),
            // Every `ue8m0` code is normal in its own terms, so its smallest normal is the bottom
            // of its range: 2^-127, written as a bit pattern because it is subnormal in f32.
            Fp8Format::UE8M0 => f32::from_bits(0x0040_0000),
        }
    }

    pub const fn subnormal_step(self) -> f32 {
        match self {
            // No subnormal ladder to step along — the codec never takes that arm for `ue8m0`.
            Fp8Format::UE8M0 => 0.0,
            _ => self.decode(1),
        }
    }

    const fn decode(self, code: u32) -> f32 {
        match self {
            Fp8Format::E4M3 => e4m3::from_bits(code as u8).to_f32(),
            Fp8Format::E5M2 => e5m2::from_bits(code as u8).to_f32(),
            // `ue8m0::to_f32` is not `const`; the code is the f32 exponent field outright.
            Fp8Format::UE8M0 => f32::from_bits(code << 23),
        }
    }

    pub fn of_type(ctx: &Context, ty: TypeHandle) -> Option<Self> {
        let ty = ty.deref(ctx);
        if ty.is::<Float8E4M3Type>() {
            Some(Fp8Format::E4M3)
        } else if ty.is::<Float8E5M2Type>() {
            Some(Fp8Format::E5M2)
        } else if ty.is::<Float8E8M0Type>() {
            Some(Fp8Format::UE8M0)
        } else {
            None
        }
    }
}
