use cubecl_ir::{Type, Value};

use crate::unexpanded;
use crate::{
    expand_assert,
    ir::{Instruction, Operator, Scope, UnaryOperands},
};
use crate::{
    expand_error,
    frontend::{CubePrimitive, CubeType},
};

use super::NativeExpand;

/// Enable elegant casting from any to any `CubeElem`
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        scope: &Scope,
        value: NativeExpand<From>,
    ) -> <Self as CubeType>::ExpandType {
        cast_expand_elem(scope, value.expand, Self::__expand_as_type(scope)).into()
    }
}

pub(crate) fn cast_expand_elem(scope: &Scope, from: Value, to_ty: Type) -> Value {
    if from.ty == to_ty {
        return from;
    }

    let vec_in = from.vector_size();
    let elems_in = vec_in * from.ty.packing_factor();
    let elems_out = to_ty.vector_size() * to_ty.packing_factor();
    if vec_in > 1 && elems_in != elems_out {
        expand_error!("Cast element count must match if input is not scalar");
    }
    let new_var = scope.create_value(to_ty.unwrap_ptr());
    scope.register(Instruction::new(
        Operator::Cast(UnaryOperands { input: from }),
        new_var,
    ));
    new_var
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
        scope: &Scope,
        value: NativeExpand<From>,
    ) -> <Self as CubeType>::ExpandType {
        let ty_in = value.expand.ty.unwrap_ptr();
        let ty_out = Self::__expand_as_type(scope).unwrap_ptr();
        let size_in = ty_in.size();
        let size_out = ty_out.size();
        expand_assert!(size_in == size_out, "Reinterpret type sizes must match");
        let new_var = scope.create_value(ty_out);
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperands {
                input: value.expand,
            }),
            new_var,
        ));
        new_var.into()
    }

    fn __expand_reinterpret_vectorization<From: CubePrimitive>(scope: &Scope) -> usize {
        let type_size = From::__expand_type_size(scope);
        type_size / Self::Scalar::__expand_type_size(scope)
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
