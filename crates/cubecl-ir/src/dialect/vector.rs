use cubecl_macros_internal::cube_op;
use pliron::r#type::TypeHandle;

use crate::{attributes::IndexAttr, interfaces::*, prelude::*, types::VectorType};

#[pliron_op(name = "vector.init", format, verifier = "succ")]
#[op_interfaces(NResultsInterface<1>, OneResultInterface, AtLeastNOpdsInterface<1>, SameOperandsType, Pure)]
pub struct VectorInitOp;
erasable!(VectorInitOp);

impl VectorInitOp {
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let value_ty = values[0].get_type(ctx);
        let out_ty = VectorType::get(ctx, value_ty, values.len());
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![out_ty.into()],
            values,
            vec![],
            0,
        );
        Self { op }
    }

    pub fn values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

#[cube_op(name = "vector.insert")]
#[result_ty(same_as = vector)]
#[op_interfaces(Pure)]
pub struct VectorInsertOp {
    pub vector: Value,
    pub value: Value,
    pub index: IndexAttr,
}
erasable!(VectorInsertOp);

#[cube_op(name = "vector.extract")]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(Pure)]
pub struct VectorExtractOp {
    pub vector: Value,
    pub index: IndexAttr,
}
erasable!(VectorExtractOp);

#[cube_op(name = "vector.insert")]
#[result_ty(same_as = vector)]
#[op_interfaces(Pure)]
pub struct VectorInsertDynamicOp {
    pub vector: Value,
    pub value: Value,
    pub index: Value,
}
erasable!(VectorInsertDynamicOp);

#[cube_op(name = "vector.extract")]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(Pure)]
pub struct VectorExtractDynamicOp {
    pub vector: Value,
    pub index: Value,
}
erasable!(VectorExtractDynamicOp);

#[cube_op(name = "vector.magnitude")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct MagnitudeOp {
    pub input: Value,
}
erasable!(MagnitudeOp);

#[cube_op(name = "vector.normalize")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct NormalizeOp {
    pub input: Value,
}
erasable!(NormalizeOp);

#[cube_op(name = "vector.sum")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct SumOp {
    pub input: Value,
}
erasable!(SumOp);

#[cube_op(name = "vector.dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct DotOp {
    pub lhs: Value,
    pub rhs: Value,
}
erasable!(DotOp);

fn scalar_ty(ctx: &Context, input: &Value) -> TypeHandle {
    input.scalar_ty(ctx)
}
