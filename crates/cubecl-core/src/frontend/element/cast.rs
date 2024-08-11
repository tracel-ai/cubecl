use crate::ir::{Item, UnaryOperator, Variable};
use crate::{frontend::ExpandElement, unexpanded};
use crate::{
    frontend::{assign, CubeContext, CubePrimitive, CubeType},
    ir::Operator,
};

/// Enable elegant casting from any to any CubeElem
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From>(
        context: &mut CubeContext,
        value: From,
    ) -> <Self as CubeType>::ExpandType
    where
        From: Into<ExpandElement>,
    {
        let value: ExpandElement = value.into();
        let var: Variable = *value;
        let new_var = context.create_local(Item::vectorized(
            <Self as CubePrimitive>::as_elem(),
            var.item().vectorization,
        ));
        assign::expand(context, value, new_var.clone());
        new_var.into()
    }
}

impl<P: CubePrimitive> Cast for P {
    fn cast_from<From: CubePrimitive>(_value: From) -> Self {
        unexpanded!()
    }
}

pub trait BitCast: CubePrimitive {
    #[allow(unused_variables)]
    fn bitcast_from<From: CubePrimitive>(value: From) -> Self {
        unexpanded!()
    }

    fn __expand_bitcast_from<From>(
        context: &mut CubeContext,
        value: From,
    ) -> <Self as CubeType>::ExpandType
    where
        From: Into<ExpandElement>,
    {
        let value: ExpandElement = value.into();
        let var: Variable = *value;
        let new_var = context.create_local(Item::vectorized(
            <Self as CubePrimitive>::as_elem(),
            var.item().vectorization,
        ));
        context.register(Operator::Bitcast(UnaryOperator {
            input: *value,
            out: *new_var.clone(),
        }));
        new_var.into()
    }
}

impl<P: CubePrimitive> BitCast for P {}
