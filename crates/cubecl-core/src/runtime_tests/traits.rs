use crate::{self as cubecl};
use cubecl::prelude::*;

#[cube]
pub(crate) trait UnaryOp: 'static + Send + Sync {
    type Options: LaunchArg;

    fn do_stuff<C: CubePrimitive>(input: C, option: Self::Options) -> C;
}

#[cube(launch)]
pub(crate) fn associated_type_input<O: UnaryOp>(_options: &O::Options) {}

pub struct Identity;

#[cube]
impl UnaryOp for Identity {
    type Options = ();

    fn do_stuff<C: CubePrimitive>(input: C, _option: Self::Options) -> C {
        input
    }
}

#[cube]
pub(crate) fn trait_as<C: CubePrimitive>(val: C) -> C {
    <Identity as UnaryOp>::do_stuff::<C>(val, ())
}
