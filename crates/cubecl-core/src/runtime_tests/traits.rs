use crate::{self as cubecl};
use cubecl::prelude::*;

#[cube]
pub(crate) trait UnaryOp<C: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn do_stuff(input: C, option: Self::Options) -> C;
}

#[cube(launch)]
pub(crate) fn associated_type_input<C: CubePrimitive, O: UnaryOp<C>>(_options: &O::Options) {}

pub struct Identity;

#[cube]
impl<C: CubePrimitive> UnaryOp<C> for Identity {
    type Options = ();

    fn do_stuff(input: C, _option: Self::Options) -> C {
        input
    }
}

#[cube]
pub(crate) fn trait_as<C: CubePrimitive>(val: C) -> C {
    <Identity as UnaryOp<C>>::do_stuff(val, ())
}
