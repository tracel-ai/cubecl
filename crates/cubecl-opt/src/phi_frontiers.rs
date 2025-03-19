use cubecl_ir::{Id, Item, Variable, VariableKind};
use petgraph::graph::NodeIndex;

use crate::{
    Optimizer,
    analyses::{dominance::DomFrontiers, liveness::Liveness, writes::Writes},
};

use super::version::{PhiEntry, PhiInstruction};

impl Optimizer {
    /// Places a phi node for each live variable at each frontier
    pub fn place_phi_nodes(&mut self) {
        let keys: Vec<_> = self.program.variables.keys().cloned().collect();
        let writes = self.analysis::<Writes>();
        let liveness = self.analysis::<Liveness>();
        let dom_frontiers = self.analysis::<DomFrontiers>();

        for var in keys {
            let mut workset: Vec<_> = self
                .node_ids()
                .iter()
                .filter(|index| writes[*index].contains(&var))
                .copied()
                .collect();
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in dom_frontiers[&node].clone() {
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
    pub fn insert_phi(&mut self, block: NodeIndex, id: Id, item: Item) {
        let var = Variable::new(VariableKind::Versioned { id, version: 0 }, item);
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
