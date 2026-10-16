use core::fmt;

use alloc::{string::String, vec::Vec};
use core::str::FromStr;
use pliron::{
    combine::{Parser, many1, satisfy},
    dict_key, input_err,
    irfmt::parsers::delimited_list_parser,
    location::Located,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

use crate::{
    ContextExt, EnumSet, FastMath, UnknownFastMath,
    dialect::math::{FAddOp, FDivOp, FMulOp, FSubOp},
    prelude::*,
};

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
                Err(err) => return input_err!(loc, "{err}")?,
            };
        }
        Ok(FastMathAttr(flags)).into_parse_result()
    }
}

/// Fast-math optimizations no op may use, whatever its kernel allows. Written `all`, or as names
/// such as `allow-contraction,not-nan`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisabledFastMath(EnumSet<FastMath>);

impl DisabledFastMath {
    pub fn new(mut optimizations: EnumSet<FastMath>) -> Self {
        // `AllowTransform` includes both, so disabling either takes it too.
        if optimizations.contains(FastMath::AllowContraction)
            || optimizations.contains(FastMath::AllowReassociation)
        {
            optimizations |= FastMath::AllowTransform;
        }
        Self(optimizations)
    }

    pub fn optimizations(self) -> EnumSet<FastMath> {
        self.0
    }

    /// What the compilation `ctx` holds disables, which is nothing unless it was told otherwise.
    pub fn of(ctx: &Context) -> Self {
        ctx.try_aux_ty::<DisabledFastMath>()
            .copied()
            .unwrap_or_default()
    }
}

impl FromStr for DisabledFastMath {
    type Err = UnknownFastMath;

    fn from_str(text: &str) -> core::result::Result<Self, Self::Err> {
        let optimizations = match text.trim() {
            "all" => FastMath::all(),
            "" => FastMath::none(),
            names => names
                .split(',')
                .map(|name| name.trim().parse())
                .collect::<core::result::Result<_, _>>()?,
        };
        Ok(Self::new(optimizations))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for DisabledFastMath {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        match self.0 == FastMath::all() {
            true => serializer.serialize_str("all"),
            false => serializer.collect_seq(self.0.iter()),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DisabledFastMath {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> core::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Written {
            Text(String),
            List(Vec<FastMath>),
        }

        match Written::deserialize(deserializer)? {
            Written::Text(text) => text.parse().map_err(serde::de::Error::custom),
            Written::List(names) => Ok(Self::new(names.into_iter().collect())),
        }
    }
}

/// A float op whose result fast-math permissions may loosen.
#[op_interface]
pub trait FastMathOp {
    verify_op_succ!();

    /// The optimizations the kernel allowed this op and the compilation did not disable, or
    /// `None` when the kernel said nothing.
    fn fast_math(&self, ctx: &Context) -> Option<EnumSet<FastMath>> {
        let allowed = self
            .get_operation()
            .get_attr::<FastMathAttr>(ctx, &ATTR_KEY_FAST_MATH)?
            .0;
        Some(allowed - DisabledFastMath::of(ctx).optimizations())
    }

    fn set_fast_math(&self, ctx: &Context, mut fast_math: EnumSet<FastMath>) {
        // Passes ask for the specific permission, and `AllowTransform` includes these two.
        if fast_math.contains(FastMath::AllowTransform) {
            fast_math |= FastMath::AllowContraction | FastMath::AllowReassociation;
        }
        self.get_operation()
            .set_attr(ctx, &ATTR_KEY_FAST_MATH, FastMathAttr(fast_math));
    }

    /// Whether this op allows every optimization in `required`. An op the kernel said nothing
    /// about allows all that are not disabled, which is how cubecl's own passes treat every op by
    /// default.
    fn allows(&self, ctx: &Context, required: EnumSet<FastMath>) -> bool {
        required.is_disjoint(DisabledFastMath::of(ctx).optimizations())
            && self
                .get_operation()
                .get_attr::<FastMathAttr>(ctx, &ATTR_KEY_FAST_MATH)
                .is_none_or(|allowed| allowed.0.is_superset(required))
    }
}

#[macro_export]
macro_rules! FastMathOp {
    ($($ty: ty),*) => {$(
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::fast_math::FastMathOp for $ty {}
    )*};
}

/// The fast-math permissions on `op`, if it is a float op the kernel said something about.
pub fn op_fast_math(ctx: &Context, op: Ptr<Operation>) -> Option<EnumSet<FastMath>> {
    op_cast::<dyn FastMathOp>(&*op.dyn_op(ctx)).and_then(|op| op.fast_math(ctx))
}

/// Whether `op` is arithmetic a downstream compiler could fuse with its neighbours, in a kernel
/// that forbids it.
pub fn forbids_contraction(ctx: &Context, op: Ptr<Operation>) -> bool {
    let fusable = op.is_op::<FAddOp>(ctx)
        || op.is_op::<FSubOp>(ctx)
        || op.is_op::<FMulOp>(ctx)
        || op.is_op::<FDivOp>(ctx);
    fusable
        && op_cast::<dyn FastMathOp>(&*op.dyn_op(ctx))
            .is_some_and(|op| !op.allows(ctx, FastMath::AllowContraction.into()))
}
