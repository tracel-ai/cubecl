use crate::{
    pliron::prelude::*,
    types::{AtomicType, PointerType},
};

#[allow(clippy::wrong_self_convention)]
pub trait OperationPtrExt: Sized {
    fn is_op<T: Op>(self, ctx: &Context) -> bool {
        self.as_op::<T>(ctx).is_some()
    }
    fn impls<T: ?Sized + OpInterfaceMarker + 'static>(self, ctx: &Context) -> bool {
        op_impls::<T>(&*self.dyn_op(ctx))
    }

    fn as_op<T: Op>(self, ctx: &Context) -> Option<T>;
    fn dyn_op(self, ctx: &Context) -> OpObj;
    fn operand(self, ctx: &Context, idx: usize) -> Value;
    fn operand_as_use(self, ctx: &Context, idx: usize) -> Use<Value>;
    fn result(self, ctx: &Context) -> Value;
}

impl OperationPtrExt for Ptr<Operation> {
    fn as_op<T: Op>(self, ctx: &Context) -> Option<T> {
        Operation::get_op(self, ctx)
    }
    fn dyn_op(self, ctx: &Context) -> OpObj {
        Operation::get_op_dyn(self, ctx)
    }
    fn operand(self, ctx: &Context, idx: usize) -> Value {
        Operation::get_operand(&self.deref(ctx), idx)
    }
    fn operand_as_use(self, ctx: &Context, idx: usize) -> Use<Value> {
        Operation::get_operand_as_use(&self.deref(ctx), idx)
    }
    fn result(self, ctx: &Context) -> Value {
        Operation::get_result(&self.deref(ctx), 0)
    }
}

macro_rules! pure_unop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[$crate::pliron::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            pub input: Value,
        }

        $crate::interfaces::erasable!($ty);
    };
}
use pliron::{
    op::{OpInterfaceMarker, OpObj, op_impls},
    value::Use,
};
pub(crate) use pure_unop;

macro_rules! pure_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = lhs)]
        #[$crate::pliron::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }

        $crate::interfaces::erasable!($ty);
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
