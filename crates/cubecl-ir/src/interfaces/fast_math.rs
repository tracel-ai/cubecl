use core::fmt;

use alloc::string::String;
use pliron::{
    combine::{Parser, many1, satisfy},
    dict_key, input_err,
    irfmt::parsers::delimited_list_parser,
    location::Located,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

use crate::{EnumSet, FastMath, prelude::*};

dict_key!(ATTR_KEY_FAST_MATH, "fast_math");

/// The float optimizations a kernel allowed one op.
#[pliron_attr(name = "cube.fast_math", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct FastMathAttr(pub EnumSet<FastMath>);

impl Printable for FastMathAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "[")?;
        for (i, flag) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{flag}")?;
        }
        write!(f, "]")
    }
}

impl Parsable for FastMathAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(input: &mut StateStream<'a>, _arg: Self::Arg) -> ParseResult<'a, Self::Parsed> {
        let loc = input.loc();
        let name = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric() || c == '-'));
        let mut list = delimited_list_parser('[', ']', ',', name);
        let names = list.parse_stream(input).into_result()?.0;

        let mut flags = FastMath::none();
        for name in names {
            match name.parse() {
                Ok(flag) => flags.insert(flag),
                Err(err) => return input_err!(loc, "{err}").into_parse_result(),
            };
        }
        Ok(FastMathAttr(flags)).into_parse_result()
    }
}

/// A float op that carries the fast-math permissions its kernel allowed.
#[op_interface]
pub trait FastMathOp {
    verify_op_succ!();

    /// The optimizations the kernel allowed this op, or `None` when the kernel said nothing.
    fn fast_math(&self, ctx: &Context) -> Option<EnumSet<FastMath>> {
        self.get_operation()
            .get_attr::<FastMathAttr>(ctx, &ATTR_KEY_FAST_MATH)
            .map(|allowed| allowed.0)
    }

    fn set_fast_math(&self, ctx: &Context, fast_math: EnumSet<FastMath>) {
        // Passes ask for the specific permission, not for one that includes it.
        let included = fast_math.iter().map(FastMath::includes);
        let fast_math = included.fold(fast_math, |all, flags| all | flags);
        self.get_operation()
            .set_attr(ctx, &ATTR_KEY_FAST_MATH, FastMathAttr(fast_math));
    }

    /// The optimizations this op may use. An op the kernel said nothing about may use all of
    /// them, which is how cubecl's own passes treat every op by default.
    fn permitted(&self, ctx: &Context) -> EnumSet<FastMath> {
        self.fast_math(ctx).unwrap_or(FastMath::all())
    }

    fn allows(&self, ctx: &Context, required: EnumSet<FastMath>) -> bool {
        self.permitted(ctx).is_superset(required)
    }

    fn allows_contraction(&self, ctx: &Context) -> bool {
        self.allows(ctx, FastMath::AllowContraction.into())
    }
}

#[macro_export]
macro_rules! FastMathOp {
    ($($ty: ty),*) => {$(
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::fast_math::FastMathOp for $ty {}
    )*};
}
