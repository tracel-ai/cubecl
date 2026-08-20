use core::fmt;

use alloc::vec::Vec;

use cubecl_macros_internal::{NamedRewrite, cube_op};
use pliron::{
    builtin::attributes::IdentifierAttr,
    combine::{Parser, parser::char::char},
    identifier::Identifier,
    input_err,
    irfmt::parsers::{process_parsed_ssa_defs, spaced, ssa_opd_parse},
    location::Location,
    op::OpObj,
    parsable::{self, IntoParseResult, Parsable, ParseResult},
    printable::{self, Printable},
};

use crate::{
    CanMaterialize,
    dialect::{
        matrix::{self, MatrixLayoutAttr, parse_closure, print_closure},
        memory,
        synchronization::SyncScope,
    },
    interfaces::{TypedExt, synchronizes},
    prelude::*,
};

/// Fill a matrix with a scalar value.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "ssa_matrix.fill")]
#[result_ty(argument)]
#[op_traits(CanMaterialize)]
pub struct FillOp {
    pub value: Value,
}

#[op_interface_impl]
impl MatrixToSSAOp for matrix::FillOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        let out_ty = self.matrix(ctx).unwrap_ptr(ctx);
        let op = FillOp::new(ctx, out_ty, value);
        let matrix = rewriter.append_op_with_result(ctx, &op);
        store_value(self.matrix(ctx), matrix, ctx, rewriter);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

#[cube_op(name = "ssa_matrix.load")]
#[result_ty(argument)]
#[op_traits(CanMaterialize)]
pub struct LoadOp {
    #[operand(ptr_read)]
    pub source: Value,
    pub stride: Value,
    pub layout: MatrixLayoutAttr,
}
synchronizes!(LoadOp, SyncScope::Plane);

#[op_interface_impl]
impl MatrixToSSAOp for matrix::LoadOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let source = self.source(ctx);
        let stride = self.stride(ctx);
        let layout = self.layout(ctx).0;
        let out_ty = self.matrix(ctx).unwrap_ptr(ctx);
        let op = LoadOp::new(ctx, out_ty, source, stride, layout);
        let matrix = rewriter.append_op_with_result(ctx, &op);
        store_value(self.matrix(ctx), matrix, ctx, rewriter);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

#[cube_op(name = "ssa_matrix.store")]
#[result_ty(none)]
#[op_traits(CanMaterialize)]
pub struct StoreOp {
    pub matrix: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub stride: Value,
    pub layout: MatrixLayoutAttr,
}
synchronizes!(StoreOp, SyncScope::Plane);

#[op_interface_impl]
impl MatrixToSSAOp for matrix::StoreOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let matrix = load_value(self.matrix(ctx), ctx, rewriter);
        let dest = self.destination(ctx);
        let stride = self.stride(ctx);
        let layout = self.layout(ctx).0;
        let op = StoreOp::new(ctx, matrix, dest, stride, layout);
        rewriter.append_op(ctx, &op);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

#[cube_op(name = "ssa_matrix.multiply_accumulate")]
#[result_ty(argument)]
#[op_traits(CanMaterialize)]
pub struct MultiplyAccumulateOp {
    pub mat_a: Value,
    pub mat_b: Value,
    pub mat_c: Value,
}
synchronizes!(MultiplyAccumulateOp, SyncScope::Plane);

#[op_interface_impl]
impl MatrixToSSAOp for matrix::MultiplyAccumulateOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let mat_a = load_value(self.mat_a(ctx), ctx, rewriter);
        let mat_b = load_value(self.mat_b(ctx), ctx, rewriter);
        let mat_c = load_value(self.mat_c(ctx), ctx, rewriter);
        let out_ty = self.mat_d(ctx).unwrap_ptr(ctx);
        let op = MultiplyAccumulateOp::new(ctx, out_ty, mat_a, mat_b, mat_c);
        let matrix_out = rewriter.append_op_with_result(ctx, &op);
        store_value(self.mat_d(ctx), matrix_out, ctx, rewriter);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

/// Cast a matrix from one type to another.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[cube_op(name = "ssa_matrix.cast")]
#[result_ty(argument)]
#[op_traits(CanMaterialize)]
pub struct CastOp {
    pub input: Value,
}

#[op_interface_impl]
impl MatrixToSSAOp for matrix::CastOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let matrix_in = load_value(self.input(ctx), ctx, rewriter);
        let out_ty = self.output(ctx).unwrap_ptr(ctx);
        let op = CastOp::new(ctx, out_ty, matrix_in);
        let matrix_out = rewriter.append_op_with_result(ctx, &op);
        store_value(self.output(ctx), matrix_out, ctx, rewriter);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

/// Executes a closure for each element in the matrix.
/// Note: Unlike most matrix ops, this does not have implicit synchronization because there's no
/// coordination between threads.
#[pliron_op(
    name = "ssa_matrix.elementwise",
    attributes = (ssa_matrix_elementwise_closure: IdentifierAttr),
    verifier = "succ"
)]
#[op_interfaces(OneResultInterface)]
#[op_traits(CanMaterialize)]
pub struct ElementwiseOp;

impl ElementwiseOp {
    pub fn new(
        ctx: &mut Context,
        matrix_in: Value,
        closure: Identifier,
        captures: Vec<Value>,
    ) -> Self {
        let out_ty = vec![matrix_in.get_type(ctx)];
        let mut opds = vec![matrix_in];
        opds.extend(captures);
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), out_ty, opds, vec![], 0),
        };
        op.set_attr_ssa_matrix_elementwise_closure(ctx, IdentifierAttr::new(closure));
        op
    }

    pub fn matrix_in(&self, ctx: &Context) -> Value {
        self.get_operation().operand(ctx, 0)
    }

    pub fn closure(&self, ctx: &Context) -> Identifier {
        let attr = self.get_attr_ssa_matrix_elementwise_closure(ctx).unwrap();
        attr.clone().into()
    }

    pub fn closure_captures(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

impl Printable for ElementwiseOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{} = {} ({}) ",
            self.get_result(ctx).disp(ctx),
            self.get_opid().disp(ctx),
            self.matrix_in(ctx).disp(ctx)
        )?;
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
        let cur_loc = input.loc();

        spaced(char('(')).parse_stream(input).into_result()?;
        let mat_in = ssa_opd_parse(input, ())?.0;
        spaced(char(')')).parse_stream(input).into_result()?;

        let (closure, captures) = parse_closure(input)?.0;
        let ctx = &mut input.state.ctx;

        if arg.len() != 1 {
            input_err!(
                cur_loc,
                "Expected 1 result, got {} during parsing",
                arg.len()
            )?;
        }

        let op = ElementwiseOp::new(ctx, mat_in, closure, captures);
        process_parsed_ssa_defs(input, &arg, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[op_interface_impl]
impl MatrixToSSAOp for matrix::ElementwiseOp {
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let matrix_in = load_value(self.matrix_in(ctx), ctx, rewriter);
        let closure = self.closure(ctx);
        let captures = self.closure_captures(ctx);

        let op = ElementwiseOp::new(ctx, matrix_in, closure, captures);
        let matrix_out = rewriter.append_op_with_result(ctx, &op);
        store_value(self.matrix_out(ctx), matrix_out, ctx, rewriter);
        rewriter.erase_operation(ctx, self.get_operation());

        Ok(())
    }
}

fn load_value(ptr: Value, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Value {
    let load = memory::LoadOp::new(ctx, ptr);
    rewriter.append_op_with_result(ctx, &load)
}

fn store_value(
    ptr: Value,
    value: Value,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) {
    let store = memory::StoreOp::new(ctx, ptr, value);
    rewriter.append_op(ctx, &store);
}

#[op_interface]
trait MatrixToSSAOp {
    verify_op_succ!();
    fn to_owned_matrix(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}

pub type MatrixToSSAPass = DialectConversionPass<MatrixToSSAConversion>;

#[derive(Default, NamedRewrite)]
pub struct MatrixToSSAConversion;

impl DialectConversion for MatrixToSSAConversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn MatrixToSSAOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let to_owned = op_cast::<dyn MatrixToSSAOp>(&*dyn_op).unwrap();
        to_owned.to_owned_matrix(ctx, rewriter, operands_info)
    }
}
