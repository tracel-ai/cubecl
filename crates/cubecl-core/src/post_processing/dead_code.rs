#![allow(unknown_lints, unnecessary_transmutes)]

use alloc::{vec, vec::Vec};
use cubecl_ir::{GlobalState, Instruction, OperationReflect};

use crate::post_processing::{
    analysis_helper::GlobalAnalyses, util::AtomicCounter, visitor::InstructionVisitor,
};

/// Eliminate non-output variables that are never read in the program.
#[derive(Debug)]
pub struct EliminateUnusedExpressions;

impl InstructionVisitor for EliminateUnusedExpressions {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        _global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        if let Some(out) = instruction.out
            && instruction.operation.is_pure()
            && !analyses.used_values().contains(&out)
        {
            changes.inc();
            vec![]
        } else {
            vec![instruction]
        }
    }
}
