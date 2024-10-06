use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, Ordering},
};

use petgraph::{graph::NodeIndex, visit::EdgeRef, Direction};

use crate::{visit_noop, Optimizer};

impl Optimizer {
    /// Do a conservative block level liveness analysis
    pub fn analyze_liveness(&mut self) {
        let vars = self.program.variables.clone();
        for var in vars.keys().copied() {
            self.analyze_var_liveness(var);
        }
    }

    fn analyze_var_liveness(&mut self, var: (u16, u8)) {
        let mut visited_edges = HashSet::new();
        let start = self.ret;
        self.analyze_block_liveness(start, var, &mut visited_edges);
    }

    fn analyze_block_liveness(
        &mut self,
        block: NodeIndex,
        var_id: (u16, u8),
        visited_edges: &mut HashSet<u32>,
    ) {
        let successors = self.sucessors(block).iter().any(|successor| {
            self.program[*successor]
                .liveness
                .get(&var_id)
                .copied()
                .unwrap_or(false)
        });
        let here = self.block_uses_var(block, var_id, successors);

        if self.program[block]
            .liveness
            .get(&var_id)
            .copied()
            .unwrap_or(false)
            != here
        {
            let incoming = self
                .program
                .edges_directed(block, Direction::Incoming)
                .map(|it| it.weight());
            for edge in incoming {
                visited_edges.remove(edge);
            }
        }

        self.program[block].liveness.insert(var_id, here);
        let edges = self
            .program
            .edges_directed(block, Direction::Incoming)
            .filter(|edge| {
                let visited = visited_edges.contains(edge.weight());
                visited_edges.insert(*edge.weight());
                !visited
            })
            .map(|edge| edge.source())
            .collect::<Vec<_>>();
        for predecessor in edges {
            self.analyze_block_liveness(predecessor, var_id, visited_edges);
        }
    }

    fn block_uses_var(
        &mut self,
        block: NodeIndex,
        var_id: (u16, u8),
        successor_live: bool,
    ) -> bool {
        let live = AtomicBool::new(successor_live);
        let ops = self.program[block].ops.clone();

        for op in ops.borrow_mut().values_mut().rev() {
            // Reads must be tracked after writes
            self.visit_operation(op, visit_noop, |opt, var| {
                if opt
                    .local_variable_id(var)
                    .map(|id| id == var_id)
                    .unwrap_or(false)
                {
                    live.store(false, Ordering::Release);
                }
            });
            self.visit_operation(
                op,
                |opt, var| {
                    if opt
                        .local_variable_id(var)
                        .map(|id| id == var_id)
                        .unwrap_or(false)
                    {
                        live.store(true, Ordering::Release);
                    }
                },
                visit_noop,
            );
        }
        live.load(Ordering::Acquire)
    }
}
