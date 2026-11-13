use core::fmt;

use alloc::string::String;
use pliron::{
    combine::{Parser, many1, satisfy},
    dict_key, input_err,
    irfmt::parsers::delimited_list_parser,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream, StreamLocation},
    printable::{self, Printable},
};

use crate::{EnumSet, FastMath, FastMathPermissions, prelude::*};

dict_key!(ATTR_KEY_FAST_MATH, "fast_math");
dict_key!(ATTR_KEY_FORBIDDEN_FAST_MATH, "forbidden_fast_math");

/// A set of float optimizations on one op: those its kernel allowed, or those it forbade.
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

    /// What the kernel gave this op.
    fn fast_math(&self, ctx: &Context) -> FastMathPermissions {
        let op = self.get_operation();
        let read = |key| op.get_attr::<FastMathAttr>(ctx, key).map(|flags| flags.0);
        FastMathPermissions {
            allowed: read(&ATTR_KEY_FAST_MATH),
            forbidden: read(&ATTR_KEY_FORBIDDEN_FAST_MATH).unwrap_or_default(),
        }
    }

    fn set_fast_math(&self, ctx: &Context, fast_math: FastMathPermissions) {
        let op = self.get_operation();
        if let Some(allowed) = fast_math.allowed {
            op.set_attr(ctx, &ATTR_KEY_FAST_MATH, FastMathAttr(allowed));
        }
        if !fast_math.forbidden.is_empty() {
            let forbidden = FastMathAttr(fast_math.forbidden);
            op.set_attr(ctx, &ATTR_KEY_FORBIDDEN_FAST_MATH, forbidden);
        }
    }

    /// The optimizations this op may use. Where the kernel said nothing, cubecl's own passes may
    /// use all of them.
    fn permitted(&self, ctx: &Context) -> EnumSet<FastMath> {
        self.fast_math(ctx).resolve(FastMath::all())
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
