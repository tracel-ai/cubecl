use cubecl_ir::{ExpandElement, Operator};

use crate::frontend::{CubePrimitive, CubeType, cast};
use crate::ir::{Instruction, Scope, Type, UnaryOperator, Variable};
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
        let line_size_in = value.expand.ty.line_size();
        let line_size_out = line_size_in * value.expand.ty.storage_type().packing_factor()
            / Self::as_type(scope).packing_factor();
        let new_var = scope
            .create_local(Type::new(<Self as CubePrimitive>::as_type(scope)).line(line_size_out));
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
        let line_size = var.ty.size() / Self::as_type(scope).size();
        let new_var = scope.create_local(
            Type::new(<Self as CubePrimitive>::as_type(scope)).line(line_size as u32),
        );
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperator { input: *value }),
            *new_var.clone(),
        ));
        new_var.into()
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
