use std::{cell::RefCell, rc::Rc};

use cubecl_ir::{Instruction, Variable};
use stable_vec::StableVec;

use crate::{ControlFlow, Optimizer, version::PhiInstruction};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockUse {
    ContinueTarget,
    Merge,
}

/// A basic block of instructions interrupted by control flow. Phi nodes are assumed to come before
/// any instructions. See <https://en.wikipedia.org/wiki/Basic_block>
#[derive(Default, Debug, Clone)]
pub struct BasicBlock {
    pub(crate) block_use: Vec<BlockUse>,
    /// The phi nodes that are required to be generated at the start of this block.
    pub phi_nodes: Rc<RefCell<Vec<PhiInstruction>>>,
    /// A stable list of operations performed in this block.
    pub ops: Rc<RefCell<StableVec<Instruction>>>,
    /// The control flow that terminates this block.
    pub control_flow: Rc<RefCell<ControlFlow>>,
}

impl Optimizer {
    /// Visit all operations in the program with the specified read and write visitors.
    pub fn visit_all(
        &mut self,
        mut visit_read: impl FnMut(&mut Self, &mut Variable) + Clone,
        mut visit_write: impl FnMut(&mut Self, &mut Variable) + Clone,
    ) {
        for node in self.program.node_indices().collect::<Vec<_>>() {
            let phi = self.program[node].phi_nodes.clone();
            let ops = self.program[node].ops.clone();
            let control_flow = self.program[node].control_flow.clone();

            for phi in phi.borrow_mut().iter_mut() {
                for elem in &mut phi.entries {
                    visit_read(self, &mut elem.value);
                }
                visit_write(self, &mut phi.out);
            }
            for op in ops.borrow_mut().values_mut() {
                self.visit_instruction(op, visit_read.clone(), visit_write.clone());
            }
            match &mut *control_flow.borrow_mut() {
                ControlFlow::IfElse { cond, .. } => visit_read(self, cond),
                ControlFlow::LoopBreak { break_cond, .. } => visit_read(self, break_cond),
                ControlFlow::Switch { value, .. } => visit_read(self, value),
                _ => {}
            };
        }
    }
}
