use std::num::NonZero;

use cubecl_ir::{ExpandElement, Operator};

use crate::frontend::{CubePrimitive, CubeType, cast};
use crate::ir::{Instruction, Item, Scope, UnaryOperator, Variable};
use crate::unexpanded;

use super::ExpandElementTyped;

/// Enable elegant casting from any to any CubeElem
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        scope: &mut Scope,
        value: ExpandElementTyped<From>,
    ) -> <Self as CubeType>::ExpandType {
        if core::any::TypeId::of::<Self>() == core::any::TypeId::of::<From>() {
            return value.expand.into();
        }
        let new_var = scope.create_local(Item::vectorized(
            <Self as CubePrimitive>::as_elem(scope),
            value.expand.item.vectorization,
        ));
        cast::expand(scope, value, new_var.clone().into());
        new_var.into()
    }
}

impl<P: CubePrimitive> Cast for P {
    fn cast_from<From: CubePrimitive>(_value: From) -> Self {
        unexpanded!()
    }
}

/// Enables reinterpetring the bits from any value to any other type of the same size.
pub trait Reinterpret: CubePrimitive {
    /// Reinterpret the bits of another primitive as this primitive without conversion.
    #[allow(unused_variables)]
    fn reinterpret<From: CubePrimitive>(value: From) -> Self {
        unexpanded!()
    }

    fn __expand_reinterpret<From: CubePrimitive>(
        scope: &mut Scope,
        value: ExpandElementTyped<From>,
    ) -> <Self as CubeType>::ExpandType {
        let value: ExpandElement = value.into();
        let var: Variable = *value;
        let vectorization = var.elem().size()
            * var
                .item
                .vectorization
                .unwrap_or(NonZero::new(1).unwrap())
                .get() as usize
            / Self::as_elem(scope).size();
        let new_var = scope.create_local(Item::vectorized(
            <Self as CubePrimitive>::as_elem(scope),
            NonZero::new(vectorization as u8),
        ));
        scope.register(Instruction::new(
            Operator::Bitcast(UnaryOperator { input: *value }),
            *new_var.clone(),
        ));
        new_var.into()
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
