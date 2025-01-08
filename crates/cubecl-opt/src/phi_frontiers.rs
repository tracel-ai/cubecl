use cubecl_core::ir::{Id, Item, Variable, VariableKind};
use petgraph::{algo::dominators::simple_fast, graph::NodeIndex, visit::EdgeRef, Direction};

use super::{
    version::{PhiEntry, PhiInstruction},
    Program,
};

impl Program {
    /// Find dominance frontiers for each block
    pub fn fill_dom_frontiers(&mut self) {
        let doms = simple_fast(&self.graph, self.root);
        for node in self.node_indices().collect::<Vec<_>>() {
            let predecessors: Vec<_> = self
                .edges_directed(node, Direction::Incoming)
                .map(|it| it.source())
                .collect();
            if predecessors.len() >= 2 {
                for predecessor in predecessors {
                    let mut runner = predecessor;
                    while runner != doms.immediate_dominator(node).unwrap() {
                        self[runner].dom_frontiers.insert(node);
                        runner = doms.immediate_dominator(runner).unwrap();
                    }
                }
            }
        }
    }

    /// Places a phi node for each live variable at each frontier
    pub fn place_phi_nodes(&mut self) {
        let keys: Vec<_> = self.variables.keys().cloned().collect();
        for var in keys {
            let mut workset: Vec<_> = self
                .node_indices()
                .filter(|index| self[*index].writes.contains(&var))
                .collect();
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in self[node].dom_frontiers.clone() {
                    if already_inserted.contains(&frontier) || self.is_dead(frontier, var) {
                        continue;
                    }
                    self.insert_phi(frontier, var, self.variables[&var]);
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
        let entries = self
            .edges_directed(block, Direction::Incoming)
            .map(|edge| edge.source())
            .map(|pred| PhiEntry {
                block: pred,
                value: var,
            });
        let phi = PhiInstruction {
            out: var,
            entries: entries.collect(),
        };
        self[block].phi_nodes.borrow_mut().push(phi);
    }
}
