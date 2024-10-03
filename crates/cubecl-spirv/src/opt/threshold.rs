use petgraph::{algo::dominators::simple_fast, visit::EdgeRef, Direction};

use super::Program;

impl Program {
    pub fn fill_dom_frontiers(&mut self) {
        let doms = simple_fast(&self.graph, self.root);
        for node in self.node_indices() {
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
        for var in self.variables.clone() {
            let mut workset: Vec<_> = self
                .node_indices()
                .filter(|index| self[*index].writes.contains(&var))
                .collect();
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in self[node].dom_frontiers.clone() {
                    if already_inserted.contains(&frontier) {
                        continue;
                    }
                    self[frontier].phi_nodes.push(var);
                    already_inserted.push(frontier);
                    if !considered.contains(&frontier) {
                        workset.push(frontier);
                        considered.push(frontier);
                    }
                }
            }
        }
    }
}
