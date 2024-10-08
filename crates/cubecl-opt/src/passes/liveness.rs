use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::NodeIndex;

use crate::{visit_noop, Optimizer};

#[derive(Clone)]
struct BlockSets {
    gen: HashSet<(u16, u8)>,
    kill: HashSet<(u16, u8)>,
}

struct State {
    worklist: VecDeque<NodeIndex>,
    block_sets: HashMap<NodeIndex, BlockSets>,
}

impl Optimizer {
    /// Do a conservative block level liveness analysis
    pub fn analyze_liveness(&mut self) {
        let mut state = State {
            worklist: VecDeque::from(self.node_ids()),
            block_sets: HashMap::new(),
        };
        while let Some(block) = state.worklist.pop_front() {
            self.analyze_block(block, &mut state);
        }
    }

    fn analyze_block(&mut self, block: NodeIndex, state: &mut State) {
        let BlockSets { gen, kill } = self.block_sets(block, state);

        let mut live_vars = gen.clone();

        for successor in self.sucessors(block) {
            let successor = &self.program[successor].live_vars;
            live_vars.extend(successor.difference(kill));
        }

        if live_vars != self.program[block].live_vars {
            state.worklist.extend(self.predecessors(block));
            self.program[block].live_vars = live_vars;
        }
    }

    fn block_sets<'a>(&mut self, block: NodeIndex, state: &'a mut State) -> &'a BlockSets {
        let block_sets = state.block_sets.entry(block);
        block_sets.or_insert_with(|| self.calculate_block_sets(block))
    }

    fn calculate_block_sets(&mut self, block: NodeIndex) -> BlockSets {
        let mut gen = HashSet::new();
        let mut kill = HashSet::new();

        let ops = self.program[block].ops.clone();

        for op in ops.borrow_mut().values_mut().rev() {
            // Reads must be tracked after writes
            self.visit_operation(op, visit_noop, |opt, var| {
                if let Some(id) = opt.local_variable_id(var) {
                    kill.insert(id);
                    gen.remove(&id);
                }
            });
            self.visit_operation(
                op,
                |opt, var| {
                    if let Some(id) = opt.local_variable_id(var) {
                        gen.insert(id);
                    }
                },
                visit_noop,
            );
        }

        BlockSets { gen, kill }
    }
}
