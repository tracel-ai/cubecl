use crate::{
    pliron::prelude::*,
    types::{AtomicType, PointerType},
};

macro_rules! pure_unop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[$crate::pliron::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            input: Value,
        }
    };
}
pub(crate) use pure_unop;

macro_rules! pure_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = lhs)]
        #[$crate::pliron::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            lhs: Value,
            rhs: Value,
        }
    };
}
pub(crate) use pure_binop;

pub(crate) fn ptr_value_ty(ctx: &Context, input: &Value) -> Ptr<TypeObj> {
    let in_ty = input.get_type(ctx).deref(ctx);
    let ptr_ty = in_ty.downcast_ref::<PointerType>();
    let mut inner = ptr_ty.expect("Should be a pointer").inner;
    // If type is atomic, it's the value inside it after load
    {
        let inner_ref = inner.deref(ctx);
        if let Some(AtomicType {
            inner: inner_val, ..
        }) = inner_ref.downcast_ref()
        {
            inner = *inner_val;
        }
    }
    inner
}
