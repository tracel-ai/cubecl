use cubecl_common::operator::Operator;

use crate::{
    ir::Elem,
    new_ir::{CubeType, NewExpr, Vectorization},
    prelude::ExpandElement,
};

macro_rules! e {
    ($ty:path) => {
        impl NewExpr<Self, Output = $ty>
    };
}

pub trait Backend: Sized {
    fn expand_binop<Left: CubeType, Right: CubeType>(
        &mut self,
        left: &e!(Left),
        right: &e!(Right),
        op: Operator,
        elem: Elem,
        vectorization: Vectorization,
    ) -> ExpandElement;
}
