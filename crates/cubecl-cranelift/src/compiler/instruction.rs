use cranelift::prelude::{
    types, Block, EntityRef, FunctionBuilder, InstBuilder, Type, Variable as CraneliftVariable,
};
use cubecl_core::{ir::Instruction as CubeInstruction, prelude::KernelDefinition};

use super::{to_ssa_type, CompilerState};

impl<'a> CompilerState<'a> {
    pub(crate) fn process_scope(&mut self, kernel: &KernelDefinition) {
        let scope_process = kernel.body.clone().process();

        scope_process.variables.iter().for_each(|var| {
            let cl_var = self.lookup.getsert_var(super::KernelVar::Local(var.kind));
            self.func_builder
                .declare_var(cl_var, to_ssa_type(&var.item));
        });
        scope_process.operations.iter().for_each(|op| {
            self.translate_instruction(op);
        });
    }
    fn translate_instruction(&mut self, op: &CubeInstruction) {
        match &op.operation {
            cubecl_core::ir::Operation::Copy(variable) => {
                if let (Some(output), Some(input)) = (&op.out, self.lookup.get(variable.kind)) {
                    todo!("Copy operation");
                }
            }
            cubecl_core::ir::Operation::Arithmetic(arithmetic) => todo!(),
            cubecl_core::ir::Operation::Comparison(comparison) => todo!(),
            cubecl_core::ir::Operation::Bitwise(bitwise) => todo!(),
            cubecl_core::ir::Operation::Operator(operator) => todo!(),
            cubecl_core::ir::Operation::Atomic(atomic_op) => todo!(),
            cubecl_core::ir::Operation::Metadata(metadata) => todo!(),
            cubecl_core::ir::Operation::Branch(branch) => todo!(),
            cubecl_core::ir::Operation::Synchronization(synchronization) => todo!(),
            cubecl_core::ir::Operation::Plane(plane) => todo!(),
            cubecl_core::ir::Operation::CoopMma(coop_mma) => todo!(),
            cubecl_core::ir::Operation::Pipeline(pipeline_ops) => todo!(),
            cubecl_core::ir::Operation::NonSemantic(non_semantic) => todo!(),
        }
    }
}
