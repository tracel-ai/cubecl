use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::{builtin::attributes::TypeAttr, derive::pliron_attr, r#type::type_cast};

use crate::{
    AddressSpace,
    attributes::{BoolAttr, IndexAttr},
    dialect::ptr_value_ty,
    interfaces::{IndexableType, Pure, erasable},
    pliron::prelude::*,
    types::PointerType,
};

#[pliron_attr(name = "memory.address_space", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash)]
pub struct AddressSpaceAttr(pub AddressSpace);

#[cube_op(name = "memory.declare_variable")]
#[result_ty(from_inputs = variable_ptr_ty)]
#[op_interfaces(Pure)]
pub struct DeclareVariableOp {
    pub value_ty: TypeAttr,
    pub addr_space: AddressSpaceAttr,
    pub alignment: IndexAttr,
}

fn variable_ptr_ty(
    ctx: &mut Context,
    value_ty: &TypeAttr,
    addr_space: &AddressSpaceAttr,
    _align: &IndexAttr,
) -> Ptr<TypeObj> {
    let value_ty = value_ty.get_type(ctx);
    PointerType::get(ctx, value_ty, addr_space.0).into()
}

#[cube_op(name = "memory.index")]
#[result_ty(from_inputs = |ctx, base, _, _, _| indexed_ptr_ty(ctx, base))]
#[op_interfaces(Pure)]
pub struct IndexOp {
    pub base: Value,
    pub index: Value,
    pub unroll_factor: IndexAttr, // Adjustment factor for bounds check
    pub checked: BoolAttr,
}
erasable!(IndexOp);

fn indexed_ptr_ty(ctx: &mut Context, base: &Value) -> Ptr<TypeObj> {
    let (value_ty, address_space) = {
        let base_ty = base.get_type(ctx).deref(ctx);
        let PointerType {
            inner,
            address_space,
        } = base_ty.downcast_ref().expect("Should be pointer");
        let list_ty = inner.deref(ctx);
        let indexable =
            type_cast::<dyn IndexableType>(list_ty.as_ref()).expect("Should be indexable");
        let value_ty = indexable.indexed_type(ctx);
        (value_ty, *address_space)
    };
    PointerType::get(ctx, value_ty, address_space).into()
}

#[cube_op(name = "memory.load")]
#[result_ty(from_inputs = ptr_value_ty)]
pub struct LoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}
erasable!(LoadOp);

#[cube_op(name = "memory.store")]
#[result_ty(none)]
pub struct StoreOp {
    #[operand(ptr_write)]
    pub ptr: Value,
    pub value: Value,
}

#[cube_op(name = "memory.copy")]
#[result_ty(none)]
pub struct CopyOp {
    #[operand(ptr_read)]
    pub source: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub len: IndexAttr,
}
