use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cubecl_core::ir::{Operation, Variable};
use petgraph::graph::NodeIndex;
use stable_vec::StableVec;

use crate::{version::PhiInstruction, ControlFlow, Optimizer, Program};

#[derive(Default, Debug, Clone)]
pub struct BasicBlock {
    pub phi_nodes: Rc<RefCell<Vec<PhiInstruction>>>,
    pub(crate) writes: HashSet<(u16, u8)>,
    pub(crate) liveness: HashMap<(u16, u8), bool>,
    pub(crate) dom_frontiers: HashSet<NodeIndex>,
    pub ops: Rc<RefCell<StableVec<Operation>>>,
    pub control_flow: Rc<RefCell<ControlFlow>>,
}

impl Optimizer {
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
                self.visit_operation(op, visit_read.clone(), visit_write.clone());
            }
            match &mut *control_flow.borrow_mut() {
                ControlFlow::If { cond, .. } => visit_read(self, cond),
                ControlFlow::IfElse { cond, .. } => visit_read(self, cond),
                ControlFlow::Switch { value, .. } => visit_read(self, value),
                _ => {}
            };
        }
    }
}

impl Program {
    pub fn is_dead(&self, node: NodeIndex, var: (u16, u8)) -> bool {
        let maybe_live = self[node].liveness.get(&var).copied().unwrap_or(false);
        !maybe_live
    }
}
