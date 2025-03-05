use cranelift::prelude::{
    types, Block, EntityRef, FunctionBuilder, Type, Variable as CraneliftVariable,
};
use cubecl_core::{
    ir::{Elem, FloatKind, IntKind, Item, UIntKind, Variable as CubeCLVariable},
    prelude::KernelDefinition,
};

use super::CompilerState;

pub fn to_ssa_type(item: &Item) -> Type {
    let mut t = match item.elem {
        Elem::Float(FloatKind::F16) => types::F16,
        Elem::Float(FloatKind::F32 | FloatKind::TF32 | FloatKind::Flex32) => types::F32,
        Elem::Float(FloatKind::F64) => types::F64,
        //https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/ir.md#integer-types
        //integer types can be interpreted as signed or unsigned.
        Elem::Int(IntKind::I8) | Elem::UInt(UIntKind::U8) | Elem::Bool => types::I8,
        Elem::Int(IntKind::I16) | Elem::UInt(UIntKind::U16) => types::I16,
        Elem::Int(IntKind::I32) | Elem::UInt(UIntKind::U32) => types::I32,
        Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64) => types::I64,

        t => panic!("Unimplemented function parameter type {:?}", t),
    };
    if let Some(size) = item.vectorization {
        t = t.by(size.get() as u32).unwrap();
    }
    t
}

impl<'a> CompilerState<'a> {
    pub(crate) fn init_variables(
        &mut self,
        builder: &mut FunctionBuilder,
        entry_block: &Block,
        kernel: &KernelDefinition,
    ) {
        let mut variables = Vec::new();
        let scope_process = kernel.body.process();
        // builder.block_params(*entry_block).iter().for_each(|val| {
        //     let var = CraneliftVariable::new(self.lookup.next_var() as usize);
        //     builder.def_var(var, *val)
        // });
        scope_process.variables.iter().for_each(|var| {
            let cranelift_variable = CraneliftVariable::new(self.lookup.next_var() as usize);
            builder.declare_var(cranelift_variable, to_ssa_type(&var.item));
            builder.def_var(cranelift_variable, val)
        });
    }
}

fn translate_variable(var: CubeCLVariable) -> CraneliftVariable {
    match var.kind {
        cubecl_core::ir::VariableKind::GlobalInputArray(Id) => todo!(),
        cubecl_core::ir::VariableKind::GlobalOutputArray(Id) => todo!(),
        cubecl_core::ir::VariableKind::GlobalScalar(Id) => todo!(),
        cubecl_core::ir::VariableKind::LocalArray { id, length } => todo!(),
        cubecl_core::ir::VariableKind::LocalMut { id } => todo!(),
        cubecl_core::ir::VariableKind::LocalConst { id } => todo!(),
        cubecl_core::ir::VariableKind::Versioned { id, version } => todo!(),
        cubecl_core::ir::VariableKind::ConstantScalar(constant_scalar_value) => todo!(),
        cubecl_core::ir::VariableKind::ConstantArray { id, length } => todo!(),
        cubecl_core::ir::VariableKind::SharedMemory { id, length } => todo!(),
        cubecl_core::ir::VariableKind::Matrix { id, mat } => todo!(),
        cubecl_core::ir::VariableKind::Slice { id } => todo!(),
        cubecl_core::ir::VariableKind::Builtin(builtin) => todo!(),
        cubecl_core::ir::VariableKind::Pipeline {
            id,
            item,
            num_stages,
        } => todo!(),
    }
}
