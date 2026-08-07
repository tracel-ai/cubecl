use alloc::string::String;
use core::{any::TypeId, fmt};
use cubecl_macros_internal::TypeHash;

use pliron::{
    builtin::type_interfaces::FloatTypeInterface,
    context::Context,
    derive::type_interface,
    parsable::{ParseResult, StateStream},
    printable,
};
use rustc_apfloat::{
    Float,
    ieee::{IeeeFloat, Semantics},
};

use crate::verify_ty_succ;

/// Type erased floating point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash)]
pub struct APFloat {
    /// Raw bits returned from `IeeeFloat<semantics>::to_bits`
    bits: u128,
    /// Marker for the semantics that can be used to guard against incorrect conversions
    semantics: TypeId,
}

impl APFloat {
    pub fn to_bits(self) -> u128 {
        self.bits
    }

    pub fn from_bits<S: 'static>(bits: u128) -> Self {
        APFloat {
            bits,
            semantics: TypeId::of::<S>(),
        }
    }

    pub fn has_semantics<S: 'static>(&self) -> bool {
        self.semantics == TypeId::of::<S>()
    }

    pub fn to_ieee<S: Semantics + 'static>(self) -> IeeeFloat<S> {
        assert_eq!(TypeId::of::<S>(), self.semantics, "Mismatched semantics");
        IeeeFloat::from_bits(self.to_bits())
    }

    pub fn from_ieee<S: Semantics + 'static>(float: IeeeFloat<S>) -> APFloat {
        Self::from_bits::<S>(float.to_bits())
    }
}

#[type_interface]
pub trait APFloatType: FloatTypeInterface {
    verify_ty_succ!();
    fn value_to_f64(&self, val: APFloat) -> f64;
    fn value_from_f64(&self, val: f64) -> APFloat;
    fn value_to_string(&self, val: APFloat) -> String;

    fn disp_value(
        &self,
        val: APFloat,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
    fn parse_value<'a>(&self, state_stream: &mut StateStream<'a>) -> ParseResult<'a, APFloat>;
}

macro_rules! apfloat_type {
    ($ty: ty, $rust_num: ty, $sem: ty) => {
        #[type_interface_impl]
        impl APFloatType for $ty {
            #[allow(unused_imports)]
            fn value_to_f64(&self, val: APFloat) -> f64 {
                num_traits::ToPrimitive::to_f64(&<$rust_num>::from_bits(
                    val.to_bits().try_into().unwrap(),
                ))
                .unwrap()
            }
            fn value_from_f64(&self, val: f64) -> APFloat {
                use rustc_apfloat::Float;
                let val: $rust_num = num_traits::NumCast::from(val).unwrap();
                APFloat::from_ieee(IeeeFloat::<$sem>::from_bits(val.to_bits().into()))
            }
            fn value_to_string(&self, val: APFloat) -> alloc::string::String {
                alloc::format!("{:#}", val.to_ieee::<$sem>())
            }
            fn disp_value(
                &self,
                val: APFloat,
                ctx: &Context,
                state: &pliron::printable::State,
                f: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                let val = val.to_ieee::<$sem>();
                let val = &val as &dyn pliron::utils::apfloat::DynFloat;
                pliron::printable::Printable::fmt(val, ctx, state, f)
            }
            fn parse_value<'a>(
                &self,
                state_stream: &mut StateStream<'a>,
            ) -> ParseResult<'a, APFloat> {
                use pliron::parsable::IntoParseResult;
                Ok(APFloat::from_ieee(
                    pliron::utils::apfloat::float_parse::<IeeeFloat<$sem>>(state_stream, ())?.0,
                ))
                .into_parse_result()
            }
        }
    };
}
pub(crate) use apfloat_type;
