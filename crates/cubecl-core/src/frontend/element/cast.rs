use cubecl_ir::{
    dialect::{
        general::{CastOp, ReinterpretCastOp},
        vector::VectorBroadcastOp,
    },
    interfaces::TypedExt,
    pliron::{r#type::Typed, value::Value},
    types::VectorType,
};
use pliron::r#type::TypeHandle;

use crate::{expand_assert, ir::Scope};
use crate::{
    expand_error,
    frontend::{CubePrimitive, CubeType},
};
use crate::{frontend::ReadValue, unexpanded};

use super::NativeExpand;

/// Enable elegant casting from any to any `CubeElem`
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: CubePrimitive>(
        scope: &Scope,
        value: NativeExpand<From>,
    ) -> <Self as CubeType>::ExpandType {
        cast_value(
            scope,
            value.read_value(scope),
            Self::__expand_as_type(scope),
        )
        .into()
    }
}

pub fn cast_value(scope: &Scope, from: Value, to_ty: TypeHandle) -> Value {
    let ctx = scope.ctx_mut();
    if from.get_type(ctx) == to_ty {
        return from;
    }

    let elems_in = from.vector_size(ctx) * from.packing_factor(ctx);
    let elems_out = to_ty.vector_size(ctx) * to_ty.packing_factor(ctx);
    if elems_in == 1 && elems_out > 1 {
        let value = broadcast_value(scope, from, elems_out);
        return cast_value(scope, value, to_ty);
    }

    if elems_in != elems_out {
        expand_error!("Cast element count must match if input is not scalar");
    }
    let op = CastOp::new(ctx, to_ty, from);
    scope.register_with_result(&op)
}

pub fn broadcast_value(scope: &Scope, value: Value, vector_size: usize) -> Value {
    if vector_size == 1 {
        return value;
    }
    let ctx = scope.ctx_mut();
    assert_eq!(value.vector_size(ctx), 1, "Can't broadcast vector");
    let vec_ty = VectorType::get(ctx, value.get_type(ctx), vector_size).to_handle();
    let op = VectorBroadcastOp::new(ctx, vec_ty, value);
    scope.register_with_result(&op)
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
        reinterpret_value(
            scope,
            value.read_value(scope),
            Self::__expand_as_type(scope),
        )
        .into()
    }

    fn __expand_reinterpret_vectorization<From: CubePrimitive>(scope: &Scope) -> usize {
        let type_size = From::__expand_size(scope);
        type_size / Self::Scalar::__expand_size(scope)
    }
}

impl<P: CubePrimitive> Reinterpret for P {}

pub fn reinterpret_value(scope: &Scope, from: Value, to_ty: TypeHandle) -> Value {
    if from.get_type(scope.ctx()) == to_ty {
        return from;
    }

    let ty_from = from.get_type(scope.ctx());
    let size_in = ty_from.size(scope.ctx());
    let size_out = to_ty.size(scope.ctx());
    expand_assert!(size_in == size_out, "Reinterpret type sizes must match");
    let op = ReinterpretCastOp::new(scope.ctx_mut(), to_ty, from);
    scope.register_with_result(&op)
}
