use crate::{
    new_ir::{self, Expr, StaticExpand, StaticExpanded},
    unexpanded,
};

use super::Primitive;

/// Enable elegant casting from any to any CubeElem
pub trait Cast<From: Primitive>: Primitive + StaticExpand
where
    <Self as StaticExpand>::Expanded: CastExpand<From, Self>,
{
    fn cast_from(value: From) -> Self;
}

pub trait CastExpand<From: Primitive, To: Primitive + Cast<From>> {
    fn cast_from(value: impl Expr<Output = From>) -> impl Expr<Output = To> {
        new_ir::Cast::new(value)
    }
}

impl<P: Primitive + StaticExpand, From: Primitive> Cast<From> for P
where
    <P as StaticExpand>::Expanded: CastExpand<From, Self>,
{
    fn cast_from(_value: From) -> Self {
        unexpanded!()
    }
}

impl<P: Primitive + StaticExpand, From: Primitive> CastExpand<From, P> for P::Expanded {}

/// Enables reinterpet-casting/bitcasting from any floating point value to any integer value and vice
/// versa
pub trait BitCast<From: Primitive>: Primitive + Sized + StaticExpand
where
    <Self as StaticExpand>::Expanded: BitCastExpand<From, Self>,
{
    const SIZE_EQUAL: () = assert!(size_of::<From>() == size_of::<Self>());
    /// Reinterpret the bits of another primitive as this primitive without conversion.
    #[allow(unused_variables)]
    fn bitcast_from(value: From) -> Self {
        unexpanded!()
    }
}

pub trait BitCastExpand<From: Primitive, To: Primitive>: Sized {
    fn bitcast_from(value: impl Expr<Output = From>) -> impl Expr<Output = To> {
        new_ir::BitCast::new(value)
    }
}

impl<From: Primitive, To: Primitive + StaticExpand> BitCast<From> for To where
    To::Expanded: BitCastExpand<From, To>
{
}
impl<From: Primitive, To: StaticExpanded> BitCastExpand<From, To::Unexpanded> for To where
    To::Unexpanded: Primitive
{
}
