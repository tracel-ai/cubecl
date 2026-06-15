use cubecl_macros_internal::cube_op;

use crate::{attributes::IndexAttr, interfaces::*, pliron::prelude::*, types::VectorType};

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
    vector: Value,
    index: Value,
    value: Value,
    #[attribute(optional)]
    const_index: IndexAttr,
}
erasable!(VectorInsertOp);

#[cube_op(name = "vector.extract")]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(Pure)]
pub struct VectorExtractOp {
    vector: Value,
    index: Value,
    #[attribute(optional)]
    const_index: IndexAttr,
}
erasable!(VectorExtractOp);

#[cube_op(name = "vector.magnitude")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct MagnitudeOp {
    input: Value,
}
erasable!(MagnitudeOp);

#[cube_op(name = "vector.normalize")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct NormalizeOp {
    input: Value,
}
erasable!(NormalizeOp);

#[cube_op(name = "vector.sum")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct SumOp {
    input: Value,
}
erasable!(SumOp);

#[cube_op(name = "vector.dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct DotOp {
    lhs: Value,
    rhs: Value,
}
erasable!(DotOp);

fn scalar_ty(ctx: &Context, input: &Value) -> Ptr<TypeObj> {
    input.scalar_ty(ctx)
}
