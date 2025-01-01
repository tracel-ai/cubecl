use cubecl_core::ir::{Item, Variable, VariableKind};
use petgraph::graph::NodeIndex;

use crate::{
    analyses::{dominators::Dominators, liveness::Liveness},
    Optimizer,
};

use super::version::{PhiEntry, PhiInstruction};

impl Optimizer {
    /// Find dominance frontiers for each block
    pub fn fill_dom_frontiers(&mut self) {
        let doms = self.analysis::<Dominators>();
        for node in self.node_ids() {
            let predecessors = self.predecessors(node);
            if predecessors.len() >= 2 {
                for predecessor in predecessors {
                    let mut runner = predecessor;
                    while runner != doms.immediate_dominator(node).unwrap() {
                        self.program[runner].dom_frontiers.insert(node);
                        runner = doms.immediate_dominator(runner).unwrap();
                    }
                }
            }
        }
    }

    /// Places a phi node for each live variable at each frontier
    pub fn place_phi_nodes(&mut self) {
        let keys: Vec<_> = self.program.variables.keys().cloned().collect();
        let liveness = self.analysis::<Liveness>();
        for var in keys {
            let mut workset: Vec<_> = self
                .node_ids()
                .iter()
                .filter(|index| self.program[**index].writes.contains(&var))
                .copied()
                .collect();
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in self.program[node].dom_frontiers.clone() {
                    if already_inserted.contains(&frontier) || liveness.is_dead(frontier, var) {
                        continue;
                    }
                    self.insert_phi(frontier, var, self.program.variables[&var]);
                    already_inserted.push(frontier);
                    if !considered.contains(&frontier) {
                        workset.push(frontier);
                        considered.push(frontier);
                    }
                }
            }
        }
    }

    /// Insert a phi node for variable `id` at `block`
    pub fn insert_phi(&mut self, block: NodeIndex, id: (u16, u8), item: Item) {
        let var = Variable::new(
            VariableKind::Versioned {
                id: id.0,
                depth: id.1,
                version: 0,
            },
            item,
        );
        let entries = self.predecessors(block).into_iter().map(|pred| PhiEntry {
            block: pred,
            value: var,
        });
        let phi = PhiInstruction {
            out: var,
            entries: entries.collect(),
        };
        self.program[block].phi_nodes.borrow_mut().push(phi);
    }
}
