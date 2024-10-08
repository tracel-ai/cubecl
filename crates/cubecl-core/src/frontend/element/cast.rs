use crate::ir::{Item, UnaryOperator, Variable};
use crate::{frontend::ExpandElement, unexpanded};
use crate::{
    frontend::{assign, CubeContext, CubePrimitive, CubeType},
    ir::Operator,
};

use super::ExpandElementTyped;

/// Enable elegant casting from any to any CubeElem
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        context: &mut CubeContext,
        value: ExpandElementTyped<From>,
    ) -> <Self as CubeType>::ExpandType {
        if core::any::TypeId::of::<Self>() == core::any::TypeId::of::<From>() {
            return value.expand.into();
        }

        let new_var = context.create_local_binding(Item::vectorized(
            <Self as CubePrimitive>::as_elem(),
            value.expand.item().vectorization,
        ));
        assign::expand(context, value, new_var.clone().into());
        new_var.into()
    }
}

impl<P: CubePrimitive> Cast for P {
    fn cast_from<From: CubePrimitive>(_value: From) -> Self {
        unexpanded!()
    }
}

/// Enables reinterpet-casting/bitcasting from any floating point value to any integer value and vice
/// versa
pub trait BitCast: CubePrimitive {
    /// Reinterpret the bits of another primitive as this primitive without conversion.
    #[allow(unused_variables)]
    fn bitcast_from<From: CubePrimitive>(value: From) -> Self {
        unexpanded!()
    }

    fn __expand_bitcast_from<From: CubePrimitive>(
        context: &mut CubeContext,
        value: ExpandElementTyped<From>,
    ) -> <Self as CubeType>::ExpandType {
        let value: ExpandElement = value.into();
        let var: Variable = *value;
        let new_var = context.create_local_binding(Item::vectorized(
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
