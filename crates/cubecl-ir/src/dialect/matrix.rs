use core::fmt;

use alloc::{string::ToString, vec::Vec};

use cubecl_macros_internal::cube_op;
use derive_more::{Deref, From};
use derive_new::new;
use itertools::Itertools;
use pliron::{
    builtin::{
        attributes::IdentifierAttr,
        types::{IntegerType, Signedness},
    },
    combine::{Parser, parser::char::char},
    derive::pliron_attr,
    identifier::Identifier,
    input_err,
    irfmt::parsers::{delimited_list_parser, spaced, ssa_opd_parse, ssa_opd_parser},
    location::Location,
    op::OpObj,
    parsable::{self, IntoParseResult, Parsable, ParseResult},
    printable::{self, Printable},
    r#type::TypedHandle,
};

use crate::{
    CanMaterialize, Pure,
    attributes::{BoolAttr, IndexAttr},
    dialect::synchronization::SyncScope,
    interfaces::{MemoryEffect, MemoryEffects, synchronizes},
    prelude::*,
    types::{
        ArrayType, MatrixShape, PointerType, VectorType,
        matrix::{MatrixLayout, MatrixType},
    },
};

#[pliron_attr(name = "matrix.layout", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord, Deref)]
pub struct MatrixLayoutAttr(pub MatrixLayout);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq, Deref)]
pub struct MatrixTypeAttr(pub TypedHandle<MatrixType>);

#[pliron_attr(name = "matrix.type", format = "$0", verifier = "succ")]
#[derive(new, From, Debug, Clone, PartialEq, Eq, Deref)]
pub struct MatrixShapeAttr(pub MatrixShape);

/// Fill a matrix with a scalar value.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.fill")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct FillOp {
    #[operand(ptr_write)]
    pub matrix: Value,
    pub value: Value,
}

#[cube_op(name = "matrix.load")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct LoadOp {
    #[operand(ptr_write)]
    pub matrix: Value,
    #[operand(ptr_read)]
    pub source: Value,
    pub stride: Value,
    pub layout: MatrixLayoutAttr,
}
synchronizes!(LoadOp, SyncScope::Plane);

#[cube_op(name = "matrix.store")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct StoreOp {
    #[operand(ptr_read)]
    pub matrix: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub stride: Value,
    pub layout: MatrixLayoutAttr,
}
synchronizes!(StoreOp, SyncScope::Plane);

#[cube_op(name = "matrix.multiply_accumulate")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct MultiplyAccumulateOp {
    pub mat_a: Value,
    pub mat_b: Value,
    pub mat_c: Value,
    #[operand(ptr_write)]
    pub mat_d: Value,
}
synchronizes!(MultiplyAccumulateOp, SyncScope::Plane);

/// Cast a matrix from one type to another.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "matrix.cast")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct CastOp {
    #[operand(ptr_read)]
    pub input: Value,
    #[operand(ptr_write)]
    pub output: Value,
}

#[cube_op(name = "matrix.row_index")]
#[result_ty(fixed = IntegerType::get(ctx, 32, Signedness::Unsigned).into())]
#[op_traits(CanMaterialize, Pure)]
pub struct RowIndexOp {
    pub lane_id: Value,
    pub i: Value,
    pub matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.col_index")]
#[result_ty(fixed = IntegerType::get(ctx, 32, Signedness::Unsigned).into())]
#[op_traits(CanMaterialize, Pure)]
pub struct ColIndexOp {
    pub lane_id: Value,
    pub i: Value,
    pub matrix_ty: MatrixTypeAttr,
}

#[cube_op(name = "matrix.ldmatrix")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct LdMatrixOp {
    pub ptr: Value,
    pub out_arr: Value,
    pub factor: IndexAttr,
    pub transpose: BoolAttr,
}
synchronizes!(LdMatrixOp, SyncScope::Plane);

impl MemoryEffects for LdMatrixOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::Read(self.ptr(ctx))]
    }
}

#[cube_op(name = "matrix.stmatrix")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
#[op_interfaces(OperandNOfType<0, ArrayType>, OperandNOfType<1, PointerType>)]
pub struct StMatrixOp {
    pub registers: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub factor: IndexAttr,
    pub transpose: BoolAttr,
}
synchronizes!(StMatrixOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
#[op_interfaces(
    OperandNOfType<0, ArrayType>, OperandNOfType<1, ArrayType>, OperandNOfType<2, ArrayType>,
    OperandNOfType<3, PointerType>,
)]
pub struct MmaManualOp {
    pub registers_a: Value,
    pub registers_b: Value,
    pub registers_c: Value,
    pub registers_d: Value,
    pub shape: MatrixShapeAttr,
}
synchronizes!(MmaManualOp, SyncScope::Plane);

#[cube_op(name = "matrix.mma_manual_scaled")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
#[op_interfaces(
    OperandNOfType<0, ArrayType>, OperandNOfType<1, ArrayType>, OperandNOfType<2, ArrayType>,
    OperandNOfType<3, PointerType>, OperandNOfType<4, VectorType>, OperandNOfType<5, VectorType>,
)]
pub struct MmaManualScaledOp {
    pub registers_a: Value,
    pub registers_b: Value,
    pub registers_c: Value,
    pub registers_d: Value,
    pub scales_a: Value,
    pub scales_b: Value,
    pub scales_factor: IndexAttr,
    pub shape: MatrixShapeAttr,
}
synchronizes!(MmaManualScaledOp, SyncScope::Plane);

/// Executes a closure for each element in the matrix.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[pliron_op(
    name = "matrix.elementwise",
    attributes = (matrix_elementwise_closure: IdentifierAttr),
    verifier = "succ"
)]
#[op_traits(CanMaterialize)]
pub struct ElementwiseOp;

impl ElementwiseOp {
    pub fn new(
        ctx: &mut Context,
        matrix_in: Value,
        matrix_out: Value,
        closure: Identifier,
        captures: Vec<Value>,
    ) -> Self {
        let mut opds = vec![matrix_in, matrix_out];
        opds.extend(captures);
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], opds, vec![], 0),
        };
        op.set_attr_matrix_elementwise_closure(ctx, IdentifierAttr::new(closure));
        op
    }

    pub fn matrix_in(&self, ctx: &Context) -> Value {
        self.get_operation().operand(ctx, 0)
    }

    pub fn matrix_out(&self, ctx: &Context) -> Value {
        self.get_operation().operand(ctx, 1)
    }

    pub fn closure(&self, ctx: &Context) -> Identifier {
        let attr = self.get_attr_matrix_elementwise_closure(ctx).unwrap();
        attr.clone().into()
    }

    pub fn closure_captures(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(2).collect()
    }
}

impl Printable for ElementwiseOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let op_id = self.get_opid().disp(ctx).to_string();
        let mat_in = self.matrix_in(ctx).disp(ctx).to_string();
        let mat_out = self.matrix_out(ctx).disp(ctx).to_string();
        write!(f, "{op_id} ({mat_in}, {mat_out}) ")?;
        print_closure(ctx, &self.closure(ctx), &self.closure_captures(ctx), f)
    }
}
impl Parsable for ElementwiseOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        input: &mut parsable::StateStream<'a>,
        arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !arg.is_empty() {
            return input_err!(input.loc(), "Expected no results").into_parse_result();
        }

        spaced(char('(')).parse_stream(input).into_result()?;
        let mat_in = ssa_opd_parse(input, ())?.0;
        spaced(char(',')).parse_stream(input).into_result()?;
        let mat_out = ssa_opd_parse(input, ())?.0;
        spaced(char(')')).parse_stream(input).into_result()?;

        let (closure, captures) = parse_closure(input)?.0;
        let ctx = &mut input.state.ctx;

        let op = ElementwiseOp::new(ctx, mat_in, mat_out, closure, captures);
        Ok(OpObj::new(op)).into_parse_result()
    }
}

/// Reusable closure printer for maybe future ops, move this if it's used elsewhere
pub fn print_closure(
    ctx: &Context,
    closure: &Identifier,
    captures: &[Value],
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let captures = captures
        .iter()
        .map(|it| it.disp(ctx).to_string())
        .join(", ");
    write!(f, "@{}({captures})", closure.disp(ctx))
}

/// Reusable closure parser for maybe future ops, move this if it's used elsewhere
pub fn parse_closure<'a>(
    input: &mut parsable::StateStream<'a>,
) -> ParseResult<'a, (Identifier, Vec<Value>)> {
    let mut parse_closure = char('@').with(Identifier::parser(()));
    let closure = parse_closure.parse_stream(input).into_result()?.0;
    let mut captures = delimited_list_parser('(', ')', ',', ssa_opd_parser());
    let captures = captures.parse_stream(input).into_result()?.0;
    Ok((closure, captures)).into_parse_result()
}
