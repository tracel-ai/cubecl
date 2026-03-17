use crate::unexpanded;
use crate::{
    expand_assert,
    ir::{Instruction, Operator, Scope, UnaryOperator},
};
use crate::{
    expand_error,
    frontend::{CubePrimitive, CubeType, cast},
};

use super::NativeExpand;

/// Enable elegant casting from any to any `CubeElem`
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        scope: &mut Scope,
        value: NativeExpand<From>,
    ) -> <Self as CubeType>::ExpandType {
        if Self::as_type(scope) == From::as_type(scope) {
            return value.expand.into();
        }
        let vec_in = value.expand.vector_size();
        let elems_in = vec_in * value.expand.ty.packing_factor();
        let elems_out = Self::__expand_vector_size(scope) * Self::__expand_packing_factor(scope);
        if vec_in > 1 && elems_in != elems_out {
            expand_error!("Cast element count must match if input is not scalar");
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

    /// Calculates the expected vectorization for the reinterpret target
    fn reinterpret_vectorization<From: CubePrimitive>() -> usize {
        unexpanded!()
    }

    fn __expand_reinterpret<From: CubePrimitive>(
        scope: &mut Scope,
        value: NativeExpand<From>,
    ) -> <Self as CubeType>::ExpandType {
        let size_in = value.expand.ty.size();
        let size_out = Self::__expand_type_size(scope);
        expand_assert!(size_in == size_out, "Reinterpret type sizes must match");
        let new_var = scope.create_local(<Self as CubePrimitive>::as_type(scope));
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperator {
                input: *value.expand,
            }),
            *new_var.clone(),
        ));
        new_var.into()
    }

    fn __expand_reinterpret_vectorization<From: CubePrimitive>(scope: &mut Scope) -> usize {
        let type_size = From::__expand_type_size(scope);
        type_size / Self::Scalar::__expand_type_size(scope)
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
