use alloc::{vec, vec::Vec};

use alloc::string::String;
use cubecl_ir::{GlobalState, Instruction, Memory, Operation, Scope};
use cubecl_runtime::server::ExecutionMode;

use crate::{
    io::*,
    post_processing::{
        analysis_helper::GlobalAnalyses, util::AtomicCounter, visitor::InstructionVisitor,
    },
};

#[derive(new, Debug)]
pub struct CheckedIoVisitor {
    mode: ExecutionMode,
    kernel_name: String,
}

impl CheckedIoVisitor {
    pub fn apply(&mut self, scope: &Scope) {
        let changes = AtomicCounter::new(0);
        // We don't care about pointer sources or used variables at this point
        let analyses = GlobalAnalyses::default();
        self.visit_scope(scope, &analyses, &changes);
    }
}

impl InstructionVisitor for CheckedIoVisitor {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        global_state: &GlobalState,
        _analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        match self.mode {
            ExecutionMode::Checked => self.transform_checked(instruction, global_state),
            ExecutionMode::Validate => self.transform_validate(instruction, global_state),
            ExecutionMode::Unchecked => vec![instruction],
        }
    }
}

impl CheckedIoVisitor {
    fn transform_checked(
        &self,
        instruction: Instruction,
        global_state: &GlobalState,
    ) -> Vec<Instruction> {
        if let Operation::Memory(memory) = &instruction.operation {
            match memory {
                Memory::Index(op) if op.checked => {
                    let has_length = op.list.has_buffer_length();

                    if has_length {
                        let list = op.list;
                        let index = op.index;
                        let scope = Scope::root(false).with_global_state(global_state.clone());

                        expand_checked_index(
                            &scope,
                            list,
                            index,
                            instruction.out(),
                            op.unroll_factor,
                        );

                        return scope.take_instructions();
                    }
                }
                _ => {}
            }
        }
        vec![instruction]
    }

    fn transform_validate(
        &self,
        instruction: Instruction,
        global_state: &GlobalState,
    ) -> Vec<Instruction> {
        if let Operation::Memory(memory) = &instruction.operation {
            match memory {
                Memory::Index(op) if op.checked => {
                    let has_length = op.list.has_buffer_length();

                    if has_length {
                        let list = op.list;
                        let index = op.index;
                        let scope = Scope::root(false).with_global_state(global_state.clone());

                        expand_validate_index(
                            &scope,
                            list,
                            index,
                            instruction.out(),
                            op.unroll_factor,
                            &self.kernel_name,
                        );

                        return scope.take_instructions();
                    }
                }
                _ => {}
            }
        }
        vec![instruction]
    }
}
