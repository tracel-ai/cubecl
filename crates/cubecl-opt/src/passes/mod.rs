mod composite;
mod dead_code;
mod disaggregate_array;
mod expression_merge;
mod index_merge;
mod inline_ref;
mod inlined_if_to_select;
mod reduce_strength;

use core::any::type_name;

pub use composite::*;
use cubecl_core::post_processing::{analysis_helper::GlobalAnalyses, visitor::InstructionVisitor};
use cubecl_ir::{Instruction, Marker};
pub use dead_code::*;
pub use disaggregate_array::*;
pub use expression_merge::*;
pub use index_merge::*;
pub use inline_ref::*;
pub use inlined_if_to_select::*;
pub use reduce_strength::*;
use stable_vec::StableVec;

use crate::{AtomicCounter, Function, GlobalState, analyses::post_order::PostOrder};

pub trait OptimizerPass {
    #[allow(unused)]
    fn apply_pre_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {}
    #[allow(unused)]
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
    }
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }
}

impl<T: InstructionVisitor> OptimizerPass for T {
    fn apply_pre_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        visit_function(self, func, state, &changes);
    }

    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        visit_function(self, func, state, &changes);
    }
}

fn visit_function<T: InstructionVisitor>(
    visitor: &mut T,
    func: &mut Function,
    state: &GlobalState,
    changes: &AtomicCounter,
) {
    let analyses = GlobalAnalyses::default();
    analyses.recalculate_pointer_source(&state.root_scope);
    analyses.recalculate_used_values(&state.root_scope);

    let blocks = func.analysis::<PostOrder>(state).reverse();

    for block in blocks {
        let instructions = core::mem::take(&mut *func[block].ops.borrow_mut());
        let mut new_instructions = StableVec::with_capacity(instructions.capacity());

        for (_, inst) in instructions {
            new_instructions.extend(visitor.visit_instruction(
                inst,
                &state.root_scope.global_state,
                &analyses,
                changes,
            ));
        }

        // Visit a fake instruction so reads can be tracked
        match &*func[block].control_flow.borrow() {
            crate::ControlFlow::IfElse { cond, .. } => {
                visitor.visit_instruction(
                    Instruction::no_out(Marker::DummyRead(*cond)),
                    &state.root_scope.global_state,
                    &analyses,
                    changes,
                );
            }
            crate::ControlFlow::Switch { value, .. } => {
                visitor.visit_instruction(
                    Instruction::no_out(Marker::DummyRead(*value)),
                    &state.root_scope.global_state,
                    &analyses,
                    changes,
                );
            }
            crate::ControlFlow::Loop { .. } => {}
            crate::ControlFlow::LoopBreak { break_cond, .. } => {
                visitor.visit_instruction(
                    Instruction::no_out(Marker::DummyRead(*break_cond)),
                    &state.root_scope.global_state,
                    &analyses,
                    changes,
                );
            }
            crate::ControlFlow::Return { value } => {
                if let Some(value) = value {
                    visitor.visit_instruction(
                        Instruction::no_out(Marker::DummyRead(*value)),
                        &state.root_scope.global_state,
                        &analyses,
                        changes,
                    );
                }
            }
            crate::ControlFlow::Unreachable => {}
            crate::ControlFlow::None => {}
        };

        *func[block].ops.borrow_mut() = new_instructions;
    }
}
