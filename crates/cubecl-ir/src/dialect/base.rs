use crate::{
    prelude::*,
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
    fn operands(self, ctx: &Context) -> Vec<Value>;
    fn operands_as_uses(self, ctx: &Context) -> Vec<Use<Value>>;
    fn result(self, ctx: &Context) -> Value;
    fn opt_result(self, ctx: &Context) -> Option<Value>;
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
    fn operands(self, ctx: &Context) -> Vec<Value> {
        self.deref(ctx).operands().collect()
    }
    fn operands_as_uses(self, ctx: &Context) -> Vec<Use<Value>> {
        self.deref(ctx).operands_as_uses().collect()
    }
    fn result(self, ctx: &Context) -> Value {
        Operation::get_result(&self.deref(ctx), 0)
    }
    fn opt_result(self, ctx: &Context) -> Option<Value> {
        self.deref(ctx).results().next()
    }
}

macro_rules! pure_unop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[$crate::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            pub input: Value,
        }

        $crate::interfaces::erasable!($ty);
        $crate::interfaces::rematerialize!($ty);
    };
}
use pliron::{
    op::{OpInterfaceMarker, OpObj, op_impls},
    r#type::TypeHandle,
    value::Use,
};
pub(crate) use pure_unop;

macro_rules! pure_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = lhs)]
        #[$crate::prelude::op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }

        $crate::interfaces::erasable!($ty);
        $crate::interfaces::rematerialize!($ty);
    };
}
pub(crate) use pure_binop;

pub fn ptr_value_ty(ctx: &Context, input: &Value) -> TypeHandle {
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
