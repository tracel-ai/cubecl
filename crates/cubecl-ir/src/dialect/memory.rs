use alloc::string::{String, ToString};
use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::{
    builtin::attributes::{TypeAttr, UnitAttr},
    derive::pliron_attr,
    printable::Printable,
    r#type::{TypeHandle, type_cast},
    verify_err,
};
use thiserror::Error;

use crate::{
    AddressSpace, CanMaterialize, NoSideEffects, Pure,
    attributes::IndexAttr,
    dialect::ptr_value_ty,
    interfaces::{IndexableType, aliasing::AliasingOp},
    prelude::*,
    types::{PointerType, scalar::IndexType},
};

#[pliron_attr(name = "memory.address_space", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct AddressSpaceAttr(pub AddressSpace);

#[cube_op(
    name = "memory.declare_variable",
    format = "attr($value_ty, $TypeAttr) `, ` attr($addr_space, $AddressSpaceAttr) `, align = ` attr($alignment, $IndexAttr)"
)]
#[result_ty(from_inputs = variable_ptr_ty)]
#[op_traits(CanMaterialize)]
pub struct DeclareVariableOp {
    pub value_ty: TypeAttr,
    pub addr_space: AddressSpaceAttr,
    pub alignment: IndexAttr,
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
        let op = Self::new(ctx, base, index);
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

#[cube_op(name = "memory.load")]
#[result_ty(from_inputs = ptr_value_ty)]
#[op_interfaces(OperandNOfType<0, PointerType>)]
#[op_traits(CanMaterialize, NoSideEffects)]
pub struct LoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
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
