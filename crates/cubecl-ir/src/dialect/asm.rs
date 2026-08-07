use core::cell::Ref;
use std::string::String;

use pliron::{
    builtin::attributes::{StringAttr, UnitAttr},
    opts::dce::SideEffects,
};

use crate::{
    CanMaterialize,
    interfaces::{MemoryEffect, MemoryEffects},
    prelude::*,
};

#[pliron_op(name = "cube.asm",
    format,
    attributes = (cube_asm_asm: StringAttr, cube_asm_pure: UnitAttr, cube_asm_nomem: UnitAttr, cube_asm_readonly: UnitAttr),
    verifier = "succ"
)]
#[op_traits(CanMaterialize)]
pub struct InlineAsmOp;

impl InlineAsmOp {
    pub fn new(
        ctx: &mut Context,
        result_types: Vec<TypeHandle>,
        asm: String,
        arguments: Vec<Value>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            arguments,
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_cube_asm_asm(ctx, asm.into());
        this
    }

    pub fn asm<'a>(&self, ctx: &'a Context) -> Ref<'a, StringAttr> {
        self.get_attr_cube_asm_asm(ctx).unwrap()
    }

    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().operands(ctx)
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn pure(&self, ctx: &Context) -> bool {
        self.get_attr_cube_asm_pure(ctx).is_some()
    }

    pub fn set_pure(&self, ctx: &Context) {
        self.set_attr_cube_asm_pure(ctx, UnitAttr::new());
    }

    pub fn nomem(&self, ctx: &Context) -> bool {
        self.get_attr_cube_asm_nomem(ctx).is_some()
    }

    pub fn set_nomem(&self, ctx: &Context) {
        self.set_attr_cube_asm_nomem(ctx, UnitAttr::new());
    }

    pub fn readonly(&self, ctx: &Context) -> bool {
        self.get_attr_cube_asm_readonly(ctx).is_some()
    }

    pub fn set_readonly(&self, ctx: &Context) {
        self.set_attr_cube_asm_readonly(ctx, UnitAttr::new());
    }
}

#[op_interface_impl]
impl SideEffects for InlineAsmOp {
    fn has_side_effects(&self, ctx: &Context) -> bool {
        !self.pure(ctx)
    }
}

#[op_interface_impl]
impl MemoryEffects for InlineAsmOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        if self.nomem(ctx) {
            vec![]
        } else if self.readonly(ctx) {
            vec![MemoryEffect::ReadAll]
        } else {
            vec![MemoryEffect::ReadAll, MemoryEffect::WriteAll]
        }
    }
}
