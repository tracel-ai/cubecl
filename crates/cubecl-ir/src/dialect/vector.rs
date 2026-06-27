use alloc::string::{String, ToString};
use cubecl_macros_internal::{cube_op, op_traits};
use pliron::{printable::Printable, r#type::TypeHandle, verify_err};
use thiserror::Error;

use crate::{
    CanMaterialize, Pure,
    attributes::IndexAttr,
    interfaces::*,
    prelude::*,
    types::{VectorType, scalar::IndexType},
};

#[pliron_op(
    name = "vector.init",
    format = "operands(CharSpace(`,`)) ` : ` type($0)"
)]
#[op_interfaces(NResultsInterface<1>, OneResultInterface, AtLeastNOpdsInterface<1>, SameOperandsType, ResultNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorInitOp;

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

#[derive(Error, Debug)]
pub enum VectorInitError {
    #[error("Output vector size doesn't match parameter count")]
    ParameterTypeMismatch,
}

impl Verify for VectorInitOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_num_operands()
            != self.get_result(ctx).vector_size(ctx)
        {
            return verify_err!(self.loc(ctx), VectorInitError::ParameterTypeMismatch)?;
        }
        Ok(())
    }
}

#[cube_op(
    name = "vector.broadcast",
    format = "$0 ` : ` type($0)",
    verifier = "custom"
)]
#[result_ty(argument)]
#[op_interfaces(ResultNOfType<0, VectorType>, TriviallyUnrollable)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorBroadcastOp {
    pub input: Value,
}

impl Verify for VectorBroadcastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let value_ty = self.input(ctx).get_type(ctx);
        let scalar_ty = self.get_result(ctx).scalar_ty(ctx);
        if scalar_ty != value_ty {
            return verify_err!(
                loc,
                VectorError::MismatchedScalarType(
                    scalar_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Index is out of range: index is {_0} but vectorization is {_1}.")]
    IndexOutOfRange(usize, usize),
    #[error("Scalar type doesn't match the inner type of the vector: expected {_0}, got {_1}")]
    MismatchedScalarType(String, String),
}

#[cube_op(
    name = "vector.insert",
    format = "$1 ` -> ` $0 `[` attr($index, $IndexAttr) `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(same_as = vector)]
#[op_interfaces(OperandNOfType<0, VectorType>, ResultNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorInsertOp {
    pub vector: Value,
    pub value: Value,
    pub index: IndexAttr,
}

impl Verify for VectorInsertOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let index = self.index(ctx).0;
        let vectorization = self.vector(ctx).vector_size(ctx);
        if index >= vectorization {
            return verify_err!(loc, VectorError::IndexOutOfRange(index, vectorization))?;
        }
        let scalar_ty = self.vector(ctx).scalar_ty(ctx);
        let value_ty = self.value(ctx).get_type(ctx);
        if scalar_ty != value_ty {
            return verify_err!(
                loc,
                VectorError::MismatchedScalarType(
                    scalar_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

#[cube_op(
    name = "vector.extract",
    format = "$0 `[` attr($index, $IndexAttr) `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorExtractOp {
    pub vector: Value,
    pub index: IndexAttr,
}

impl Verify for VectorExtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let index = self.index(ctx).0;
        let vectorization = self.vector(ctx).vector_size(ctx);
        if index >= vectorization {
            return verify_err!(loc, VectorError::IndexOutOfRange(index, vectorization))?;
        }
        Ok(())
    }
}

#[cube_op(
    name = "vector.insert_dynamic",
    format = "$1 ` -> ` $0 `[` $2 `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(same_as = vector)]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<2, IndexType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorInsertDynamicOp {
    pub vector: Value,
    pub value: Value,
    pub index: Value,
}

impl Verify for VectorInsertDynamicOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let scalar_ty = self.vector(ctx).scalar_ty(ctx);
        let value_ty = self.value(ctx).get_type(ctx);
        if scalar_ty != value_ty {
            return verify_err!(
                loc,
                VectorError::MismatchedScalarType(
                    scalar_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

#[cube_op(name = "vector.extract_dynamic", format = "$0 `[` $1 `] : ` type($0)")]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, IndexType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorExtractDynamicOp {
    pub vector: Value,
    pub index: Value,
}

#[cube_op(name = "vector.magnitude")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct MagnitudeOp {
    pub input: Value,
}

#[cube_op(name = "vector.normalize")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct NormalizeOp {
    pub input: Value,
}

#[cube_op(name = "vector.sum")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct SumOp {
    pub input: Value,
}

#[cube_op(name = "vector.dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct DotOp {
    pub lhs: Value,
    pub rhs: Value,
}

fn scalar_ty(ctx: &Context, input: &Value) -> TypeHandle {
    input.scalar_ty(ctx)
}
