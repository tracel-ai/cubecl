use cubecl_macros_internal::cube_op;

use crate::{
    dialect::base::pure_binop,
    interfaces::{Pure, TypedExt},
    pliron::prelude::*,
    types::{VectorType, scalar::BoolType},
};

pure_binop!("cmp.min", MinOp);
pure_binop!("cmp.max", MaxOp);

#[cube_op(name = "cmp.clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct ClampOp {
    input: Value,
    min: Value,
    max: Value,
}

macro_rules! cmp_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(from_inputs = cmp_result_ty)]
        #[$crate::pliron::prelude::op_interfaces(SameOperandsType, Pure)]
        pub struct $ty {
            lhs: Value,
            rhs: Value,
        }
    };
}

cmp_binop!("cmp.less_than", LessThanOp);
cmp_binop!("cmp.greater_than", GreaterThanOp);
cmp_binop!("cmp.less_than_or_equal", LessThanOrEqualOp);
cmp_binop!("cmp.greater_than_or_equal", GreaterThanOrEqualOp);
cmp_binop!("cmp.equal", EqualOp);
cmp_binop!("cmp.not_equal", NotEqualOp);

fn cmp_result_ty(ctx: &mut Context, lhs: &Value, _: &Value) -> Ptr<TypeObj> {
    let vectorization = lhs.vector_size(ctx);
    let bool = BoolType::get(ctx).into();
    if vectorization == 1 {
        bool
    } else {
        VectorType::get(ctx, bool, vectorization).into()
    }
}
