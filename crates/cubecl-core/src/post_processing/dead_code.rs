#![allow(unknown_lints, unnecessary_transmutes)]

use alloc::{vec, vec::Vec};
use cubecl_ir::{GlobalState, Instruction, OperationReflect, Scope, Variable};
use hashbrown::HashSet;

use crate::post_processing::{
    util::AtomicCounter,
    visitor::{InstructionVisitor, Visitor, visit_scope},
};

/// Eliminate non-output variables that are never read in the program.
#[derive(Debug)]
pub struct EliminateUnusedExpressions;

#[derive(Default, Debug)]
struct CollectUses {
    used_variables: HashSet<Variable>,
}

#[derive(Debug)]
struct EliminateUnused {
    used_variables: HashSet<Variable>,
}

impl InstructionVisitor for EliminateUnusedExpressions {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        _global_state: &GlobalState,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        vec![instruction]
    }

    fn visit_scope(&mut self, scope: &Scope, changes: &AtomicCounter) {
        let mut uses = CollectUses::default();
        visit_scope(&mut uses, scope, changes);
        let mut eliminate = EliminateUnused {
            used_variables: uses.used_variables,
        };
        visit_scope(&mut eliminate, scope, changes);
        let state = scope.state();
        let mut locals = state.allocator.local_mut_pool.borrow_mut();
        locals.retain(|it| eliminate.used_variables.contains(it));
    }
}

impl InstructionVisitor for CollectUses {
    fn visit_instruction(
        &mut self,
        mut instruction: Instruction,
        _global_state: &GlobalState,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(self);
        visitor.visit_operation(&mut instruction.operation, |this, var| {
            this.used_variables.insert(*var);
        });
        vec![instruction]
    }
}

impl InstructionVisitor for EliminateUnused {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        _global_state: &GlobalState,
        changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        if let Some(out) = instruction.out
            && instruction.operation.is_pure()
            && !self.used_variables.contains(&out)
        {
            changes.inc();
            vec![]
        } else {
            vec![instruction]
        }
    }
}
