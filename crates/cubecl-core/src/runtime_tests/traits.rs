use crate::{self as cubecl};
use cubecl::prelude::*;

#[cube]
pub(crate) trait UnaryOp<C: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;
}

#[cube(launch)]
pub(crate) fn associated_type_input<C: CubePrimitive, O: UnaryOp<C>>(_options: &O::Options) {}
