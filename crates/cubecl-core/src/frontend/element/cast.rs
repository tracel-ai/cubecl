use cubecl_ir::{ExpandElement, Operator};

use crate::frontend::{CubePrimitive, CubeType, cast};
use crate::ir::{Instruction, Scope, UnaryOperator};
use crate::unexpanded;

use super::ExpandElementTyped;

/// Enable elegant casting from any to any `CubeElem`
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        scope: &mut Scope,
        value: ExpandElementTyped<From>,
    ) -> <Self as CubeType>::ExpandType {
        if Self::as_type(scope) == From::as_type(scope) {
            return value.expand.into();
        }
        let new_var = scope.create_local(<Self as CubePrimitive>::as_type(scope));
        cast::expand::<From, Self>(scope, value, new_var.clone().into());
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
        let new_var = scope.create_local(<Self as CubePrimitive>::as_type(scope));
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperator { input: *value }),
            *new_var.clone(),
        ));
        new_var.into()
    }
}

#[allow(unused)]
pub fn reinterpret_line_size<From: CubePrimitive, To: CubePrimitive>(value: &From) -> usize {
    unexpanded!()
}

pub mod reinterpret_line_size {
    use super::*;

    pub fn expand<From: CubePrimitive, To: CubePrimitive>(
        scope: &mut Scope,
        value: ExpandElementTyped<From>,
    ) -> usize {
        let type_size = From::__expand_type_size(scope) * value.expand.line_size();
        type_size / To::__expand_type_size(scope)
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
