use cubecl_ir::{
    dialect::general::{CastOp, ReinterpretCastOp},
    interfaces::TypedExt,
    pliron::{
        builtin::op_interfaces::OneResultInterface,
        context::Ptr,
        r#type::{TypeObj, Typed},
        value::Value,
    },
};

use crate::unexpanded;
use crate::{expand_assert, ir::Scope};
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
        cast_value(scope, value.value(scope), Self::__expand_as_type(scope)).into()
    }
}

pub(crate) fn cast_value(scope: &Scope, from: Value, to_ty: Ptr<TypeObj>) -> Value {
    if from.get_type(&scope.ctx()) == to_ty {
        return from;
    }

    let vec_in = from.vector_size(&scope.ctx());
    let elems_in = vec_in * from.packing_factor(&scope.ctx());
    let elems_out = to_ty.vector_size(&scope.ctx()) * to_ty.packing_factor(&scope.ctx());
    if vec_in > 1 && elems_in != elems_out {
        expand_error!("Cast element count must match if input is not scalar");
    }
    let op = CastOp::new(&mut scope.ctx_mut(), to_ty, from);
    scope.register(&op);
    op.get_result(&scope.ctx())
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
        let value = value.read_value(scope);
        let ty_in = value.get_type(&scope.ctx());
        let ty_out = Self::__expand_as_type(scope);
        let size_in = ty_in.size(&scope.ctx());
        let size_out = ty_out.size(&scope.ctx());
        expand_assert!(size_in == size_out, "Reinterpret type sizes must match");
        let op = ReinterpretCastOp::new(&mut scope.ctx_mut(), ty_out, value);
        scope.register(&op);
        op.get_result(&scope.ctx()).into()
    }

    fn __expand_reinterpret_vectorization<From: CubePrimitive>(scope: &Scope) -> usize {
        let type_size = From::__expand_size(scope);
        type_size / Self::Scalar::__expand_size(scope)
    }
}

impl<P: CubePrimitive> Reinterpret for P {}
