use ::pliron::parsable::ParseResult;
use alloc::string::{String, ToString};
use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::{
    arg_err,
    attribute::AttrObj,
    builtin::{
        attributes::{TypeAttr, UnitAttr},
        ops::ConstantOp,
    },
    combine::{
        Parser, optional,
        parser::char::{char, spaces, string},
    },
    derive::pliron_attr,
    identifier::Identifier,
    input_err,
    irbuild::inserter::Inserter,
    irfmt::parsers::{process_parsed_ssa_defs, spaced},
    location::Location,
    op::{OpBox, OpObj},
    opts::mem2reg::{
        AllocInfo, PromotableAllocationInterface, PromotableOpInterface, PromotableOpKind,
    },
    parsable::{IntoParseResult, Parsable},
    printable::Printable,
    r#type::{TypeHandle, type_cast},
    verify_err,
};
use thiserror::Error;

use crate::{
    AddressSpace, CanMaterialize, NoSideEffects, Pure,
    attributes::IndexAttr,
    dialect::{general::PoisonOp, ptr_value_ty},
    interfaces::{IndexableType, aliasing::AliasingOp},
    prelude::*,
    types::{PointerType, scalar::IndexType},
};

#[pliron_attr(name = "memory.address_space", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct AddressSpaceAttr(pub AddressSpace);

#[cube_op(name = "memory.declare_variable", format = "custom")]
#[result_ty(from_inputs = variable_ptr_ty)]
#[op_traits(NoSideEffects, CanMaterialize)]
pub struct DeclareVariableOp {
    pub value_ty: TypeAttr,
    pub addr_space: AddressSpaceAttr,
    pub alignment: IndexAttr,
    #[attribute(optional, untyped)]
    pub initializer: AttrObj,
}

impl Printable for DeclareVariableOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} = {} {} {}, align = {}",
            self.get_result(ctx).disp(ctx),
            self.get_opid(),
            self.value_ty(ctx).disp(ctx),
            self.addr_space(ctx).disp(ctx),
            self.alignment(ctx).disp(ctx)
        )?;
        if let Some(init) = self.initializer(ctx) {
            write!(f, ", init = {}", init.disp(ctx))?;
        }

        Ok(())
    }
}
impl Parsable for DeclareVariableOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        input: &mut ::pliron::parsable::StateStream<'a>,
        arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let cur_loc = input.loc();
        let value_ty = TypeAttr::parse(input, ())?.0;
        spaces().parse_stream(input).into_result()?;
        let addr_space = AddressSpaceAttr::parse(input, ())?.0;
        let mut label = (spaced(char(',')), string("align"), spaced(char('=')));
        label.parse_stream(input).into_result()?;
        let align = IndexAttr::parse(input, ())?.0;
        let mut label = (spaced(char(',')), string("init"), spaced(char('=')));
        label.parse_stream(input).into_result()?;
        let mut init_parse = optional(AttrObj::parser(()));
        let init = init_parse.parse_stream(input).into_result()?.0;

        let ctx = &mut input.state.ctx;
        if arg.len() != 1 {
            return input_err!(
                cur_loc,
                "Expected 1 result, got {} during parsing",
                arg.len()
            )?;
        }
        let op = DeclareVariableOp::new(ctx, value_ty, addr_space, align, init);
        process_parsed_ssa_defs(input, &arg, op.get_operation())?;
        Ok(OpBox::new(op)).into_parse_result()
    }
}

#[op_interface_impl]
impl PromotableAllocationInterface for DeclareVariableOp {
    fn alloc_info(&self, ctx: &Context) -> Vec<AllocInfo> {
        if self.addr_space(ctx).0 == AddressSpace::Local {
            vec![AllocInfo {
                ptr: self.get_result(ctx),
                ty: self.value_ty(ctx).get_type(ctx),
            }]
        } else {
            vec![]
        }
    }

    fn default_value(
        &self,
        ctx: &mut Context,
        inserter: &mut dyn Inserter,
        alloc_info: &AllocInfo,
    ) -> Result<Value> {
        if alloc_info.ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        if let Some(initializer) = self.initializer(ctx).map(|it| it.clone()) {
            let constant = ConstantOp::new(ctx, initializer);
            inserter.insert_op(ctx, &constant);
            Ok(constant.get_result(ctx))
        } else {
            let poison = PoisonOp::new(ctx, alloc_info.ty);
            inserter.insert_op(ctx, &poison);
            Ok(poison.get_result(ctx))
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        rewriter: &mut dyn Rewriter,
        alloc_infos: &[AllocInfo],
    ) -> Result<()> {
        if alloc_infos.len() != 1 || alloc_infos[0].ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

fn variable_ptr_ty(
    ctx: &Context,
    value_ty: &TypeAttr,
    addr_space: &AddressSpaceAttr,
    _align: &IndexAttr,
) -> TypeHandle {
    let value_ty = value_ty.get_type(ctx);
    PointerType::get(ctx, value_ty, addr_space.0).into()
}

#[cube_op(
    name = "memory.index",
    format = "$0 `[` $1 `]` opt_attr($checked, $UnitAttr) ` : ` type($0)"
)]
#[result_ty(from_inputs = |ctx, base, _| indexed_ptr_ty(ctx, base))]
#[op_interfaces(OperandNOfType<0, PointerType>, OperandNOfType<1, IndexType>)]
#[op_traits(Pure, CanMaterialize)]
pub struct IndexOp {
    pub base: Value,
    pub index: Value,
    #[attribute(optional)]
    pub checked: UnitAttr,
}

#[op_interface_impl]
impl AliasingOp for IndexOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        Some(self.base(ctx))
    }
}

impl IndexOp {
    pub fn maybe_checked(ctx: &mut Context, base: Value, index: Value, checked: bool) -> Self {
        let op = Self::new(ctx, base, index, checked.then_some(UnitAttr::new()));
        if checked {
            op.set_checked(ctx);
        }
        op
    }
}

fn indexed_ptr_ty(ctx: &Context, base: &Value) -> TypeHandle {
    let (value_ty, address_space) = {
        let base_ty = base.get_type(ctx).deref(ctx);
        let PointerType {
            inner,
            address_space,
        } = base_ty.downcast_ref().expect("Should be pointer");
        let list_ty = inner.deref(ctx);
        let indexable = type_cast::<dyn IndexableType>(&*list_ty).expect("Should be indexable");
        let value_ty = indexable.indexed_type(ctx);
        (value_ty, *address_space)
    };
    PointerType::get(ctx, value_ty, address_space).into()
}

#[derive(Error, Debug)]
#[error("Register Promotion: Allocation info provided is not related to this operation")]
pub struct UnrelatedAllocInfo;

#[cube_op(name = "memory.load")]
#[result_ty(from_inputs = ptr_value_ty)]
#[op_interfaces(OperandNOfType<0, PointerType>)]
#[op_traits(CanMaterialize, NoSideEffects)]
pub struct LoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}

#[op_interface_impl]
impl PromotableOpInterface for LoadOp {
    fn promotion_kind(&self, ctx: &Context, alloc_info: &AllocInfo) -> PromotableOpKind {
        if self.ptr(ctx) == alloc_info.ptr {
            PromotableOpKind::Load
        } else {
            PromotableOpKind::NonPromotableUse
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        alloc_info_reaching_defs: &[(AllocInfo, Value)],
        rewriter: &mut dyn Rewriter,
    ) -> Result<()> {
        if alloc_info_reaching_defs.len() != 1 {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        let (alloc_info, reaching_def) = &alloc_info_reaching_defs[0];
        if self.ptr(ctx) != alloc_info.ptr {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![*reaching_def]);
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("Value type doesn't match the inner type of the pointer: expected {_0}, got {_1}")]
    MismatchedValueType(String, String),
}

#[cube_op(name = "memory.store", verifier = "custom")]
#[result_ty(none)]
#[op_interfaces(OperandNOfType<0, PointerType>)]
#[op_traits(CanMaterialize)]
pub struct StoreOp {
    #[operand(ptr_write)]
    pub ptr: Value,
    pub value: Value,
}

impl Verify for StoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let ptr_value_ty = ptr_value_ty(ctx, &self.ptr(ctx));
        let value_ty = self.value(ctx).get_type(ctx);
        if ptr_value_ty != value_ty {
            return verify_err!(
                loc,
                StoreError::MismatchedValueType(
                    ptr_value_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

#[op_interface_impl]
impl PromotableOpInterface for StoreOp {
    fn promotion_kind(&self, ctx: &Context, alloc_info: &AllocInfo) -> PromotableOpKind {
        if self.ptr(ctx) == alloc_info.ptr {
            PromotableOpKind::Store(self.value(ctx))
        } else {
            PromotableOpKind::NonPromotableUse
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        alloc_info_reaching_defs: &[(AllocInfo, Value)],
        rewriter: &mut dyn Rewriter,
    ) -> Result<()> {
        if alloc_info_reaching_defs.len() != 1 {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        let (alloc_info, _reaching_def) = &alloc_info_reaching_defs[0];
        if self.ptr(ctx) != alloc_info.ptr {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[cube_op(name = "memory.copy")]
#[result_ty(none)]
#[op_interfaces(OperandNOfType<0, PointerType>, SameOperandsType)]
#[op_traits(CanMaterialize)]
pub struct CopyOp {
    #[operand(ptr_read)]
    pub source: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub len: IndexAttr,
}
