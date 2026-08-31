use core::cell::Ref;
use std::string::String;

use derive_new::new;
use pliron::{
    builtin::attributes::{StringAttr, UnitAttr},
    opts::dce::SideEffects,
};

use crate::{
    AddressSpace, AddressSpaceVecAttr, CanMaterialize,
    interfaces::{MemoryEffect, MemoryEffects},
    prelude::*,
    typed_vec_attr,
};

#[format]
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum InputKind {
    /// Regular input
    In,
    /// Input that reads memory through the value
    MemIn,
    /// Input that writes memory through the value.
    MemOut,
    /// Input that reads and writes memory through the value.
    MemInout,
}

#[pliron_attr(name = "asm.reg_class", format, verifier = "succ")]
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum RegSpec {
    /// Default register class for the type
    Inferred,
    /// Custom class mainly for CPU (i.e. `xmm_reg`)
    Class(String),
    /// Explicit register mainly for CPU (i.e. `"ax"`)
    Explicit(String),
}

#[pliron_attr(name = "asm.input_spec", format, verifier = "succ")]
#[derive(PartialEq, Eq, Hash, Clone, Debug, new)]
pub struct InputSpec {
    pub kind: InputKind,
    pub class: RegSpec,
}

typed_vec_attr!(RegSpec, "asm.reg_specs");
typed_vec_attr!(InputSpec, "asm.input_specs");

#[pliron_op(name = "cube.asm",
    format,
    attributes = (
        cube_asm_asm: StringAttr,
        cube_asm_pure: UnitAttr,
        cube_asm_nomem: UnitAttr,
        cube_asm_explicit_mem: UnitAttr,
        cube_asm_readonly: UnitAttr,
        cube_asm_out_spec: RegSpecVecAttr,
        cube_asm_in_spec: InputSpecVecAttr,
        cube_asm_reads_spaces: AddressSpaceVecAttr,
        cube_asm_writes_spaces: AddressSpaceVecAttr,
    ),
    verifier = "succ"
)]
#[op_traits(CanMaterialize)]
pub struct InlineAsmOp;

impl InlineAsmOp {
    pub fn new(
        ctx: &mut Context,
        result_types: Vec<TypeHandle>,
        out_spec: Vec<RegSpec>,
        asm: String,
        arguments: Vec<Value>,
        in_spec: Vec<InputSpec>,
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
        this.set_attr_cube_asm_out_spec(ctx, out_spec.into());
        this.set_attr_cube_asm_in_spec(ctx, in_spec.into());
        this.set_attr_cube_asm_reads_spaces(ctx, Default::default());
        this.set_attr_cube_asm_writes_spaces(ctx, Default::default());
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

    pub fn explicit_mem(&self, ctx: &Context) -> bool {
        self.get_attr_cube_asm_explicit_mem(ctx).is_some()
    }

    pub fn set_explicit_mem(&self, ctx: &Context) {
        self.set_attr_cube_asm_explicit_mem(ctx, UnitAttr::new());
    }

    pub fn readonly(&self, ctx: &Context) -> bool {
        self.get_attr_cube_asm_readonly(ctx).is_some()
    }

    pub fn set_readonly(&self, ctx: &Context) {
        self.set_attr_cube_asm_readonly(ctx, UnitAttr::new());
    }

    pub fn reads_spaces(&self, ctx: &Context) -> Vec<AddressSpace> {
        self.get_attr_cube_asm_reads_spaces(ctx).unwrap().0.clone()
    }

    pub fn set_reads_spaces(&self, ctx: &Context, spaces: Vec<AddressSpace>) {
        self.set_attr_cube_asm_reads_spaces(ctx, spaces.into());
    }

    pub fn writes_spaces(&self, ctx: &Context) -> Vec<AddressSpace> {
        self.get_attr_cube_asm_writes_spaces(ctx).unwrap().0.clone()
    }

    pub fn set_writes_spaces(&self, ctx: &Context, spaces: Vec<AddressSpace>) {
        self.set_attr_cube_asm_writes_spaces(ctx, spaces.into());
    }

    pub fn out_specs(&self, ctx: &Context) -> Vec<RegSpec> {
        self.get_attr_cube_asm_out_spec(ctx).unwrap().0.clone()
    }

    pub fn in_specs(&self, ctx: &Context) -> Vec<InputSpec> {
        self.get_attr_cube_asm_in_spec(ctx).unwrap().0.clone()
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
        } else if self.explicit_mem(ctx) {
            let mut out = vec![];
            for space in self.reads_spaces(ctx) {
                out.push(MemoryEffect::ReadAllInSpace(space));
            }
            for space in self.writes_spaces(ctx) {
                out.push(MemoryEffect::WriteAllInSpace(space));
            }
            for (value, spec) in self.inputs(ctx).into_iter().zip(self.in_specs(ctx)) {
                match spec.kind {
                    InputKind::MemIn => {
                        out.push(MemoryEffect::Read(value));
                    }
                    InputKind::MemOut => {
                        out.push(MemoryEffect::Write(value));
                    }
                    InputKind::MemInout => {
                        out.push(MemoryEffect::Read(value));
                        out.push(MemoryEffect::Write(value));
                    }
                    InputKind::In => {}
                }
            }
            out
        } else {
            vec![MemoryEffect::ReadAll, MemoryEffect::WriteAll]
        }
    }
}
