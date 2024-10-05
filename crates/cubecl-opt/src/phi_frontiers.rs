use cubecl_core::ir::Item;
use petgraph::{algo::dominators::simple_fast, graph::NodeIndex, visit::EdgeRef, Direction};

use super::{
    version::{PhiEntry, PhiInstruction},
    Program,
};

impl Program {
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

    pub fn place_phi_nodes(&mut self) {
        let keys: Vec<_> = self.variables.keys().cloned().collect();
        for var in keys {
            let mut workset: Vec<_> = self
                .node_indices()
                .filter(|index| self[*index].writes.contains(&var))
                .collect();
            println!("var {var:?} is written to in {workset:?}");
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in self[node].dom_frontiers.clone() {
                    if already_inserted.contains(&frontier) {
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

    pub fn insert_phi(&mut self, block: NodeIndex, id: (u16, u8), item: Item) {
        let out_id = (id.0, id.1, 0);
        let entries = self
            .edges_directed(block, Direction::Incoming)
            .map(|edge| edge.source())
            .map(|pred| PhiEntry {
                block: pred,
                value: (id.0, id.1, 0),
            });
        let phi = PhiInstruction {
            out: out_id,
            entries: entries.collect(),
            item,
        };
        self[block].phi_nodes.push(phi);
    }
}
