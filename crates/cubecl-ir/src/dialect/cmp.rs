use cubecl_macros_internal::{const_eval, cube_op};
use half::{bf16, f16};
use pliron::r#type::TypeHandle;

use crate::{
    attributes::{BoolAttr, FloatAttr, IndexAttr, IntAttr, UIntAttr},
    dialect::base::pure_binop,
    interfaces::{Pure, TypedExt, erasable},
    prelude::*,
    types::{VectorType, scalar::BoolType},
};

pure_binop!("cmp.min", MinOp);
const_eval!(MinOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.min(rhs)
});

pure_binop!("cmp.max", MaxOp);
const_eval!(MaxOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.max(rhs)
});

#[cube_op(name = "cmp.clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct ClampOp {
    pub input: Value,
    pub min: Value,
    pub max: Value,
}
erasable!(ClampOp);
const_eval!(ClampOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |inp, min, max| inp.clamp(min, max)
});

macro_rules! cmp_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(from_inputs = cmp_result_ty)]
        #[$crate::prelude::op_interfaces(SameOperandsType, Pure)]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }

        $crate::interfaces::erasable!($ty);
    };
}

cmp_binop!("cmp.less_than", LessThanOp);
const_eval!(LessThanOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs < rhs).into()
    }
});

cmp_binop!("cmp.greater_than", GreaterThanOp);
const_eval!(GreaterThanOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs > rhs).into()
    }
});

cmp_binop!("cmp.less_than_or_equal", LessThanOrEqualOp);
const_eval!(LessThanOrEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs <= rhs).into()
    }
});

cmp_binop!("cmp.greater_than_or_equal", GreaterThanOrEqualOp);
const_eval!(GreaterThanOrEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs >= rhs).into()
    }
});

cmp_binop!("cmp.equal", EqualOp);
const_eval!(EqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs == rhs).into()
    }
});

cmp_binop!("cmp.not_equal", NotEqualOp);
const_eval!(NotEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs != rhs).into()
    }
});

fn cmp_result_ty(ctx: &Context, lhs: &Value, _: &Value) -> TypeHandle {
    let vectorization = lhs.vector_size(ctx);
    let bool = BoolType::get(ctx).into();
    if vectorization == 1 {
        bool
    } else {
        VectorType::get(ctx, bool, vectorization).into()
    }
}
