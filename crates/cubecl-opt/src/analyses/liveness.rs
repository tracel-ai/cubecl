use std::collections::{HashMap, HashSet, VecDeque};

use cubecl_ir::Id;
use petgraph::graph::NodeIndex;

use crate::{Optimizer, analyses::post_order::PostOrder};

use super::Analysis;

pub struct Liveness {
    live_vars: HashMap<NodeIndex, HashSet<Id>>,
}

#[derive(Clone)]
struct BlockSets {
    generated: HashSet<Id>,
    kill: HashSet<Id>,
}

struct State {
    worklist: VecDeque<NodeIndex>,
    block_sets: HashMap<NodeIndex, BlockSets>,
}

impl Analysis for Liveness {
    fn init(opt: &mut Optimizer) -> Self {
        let mut this = Self::empty(opt);
        this.analyze_liveness(opt);
        this
    }
}

impl Liveness {
    pub fn empty(opt: &Optimizer) -> Self {
        let live_vars = opt
            .node_ids()
            .iter()
            .map(|it| (*it, HashSet::new()))
            .collect();
        Self { live_vars }
    }

    pub fn at_block(&self, block: NodeIndex) -> &HashSet<Id> {
        &self.live_vars[&block]
    }

    pub fn is_dead(&self, node: NodeIndex, var: Id) -> bool {
        !self.at_block(node).contains(&var)
    }

    /// Do a conservative block level liveness analysis
    pub fn analyze_liveness(&mut self, opt: &mut Optimizer) {
        let mut state = State {
            worklist: VecDeque::from(opt.analysis::<PostOrder>().forward()),
            block_sets: HashMap::new(),
        };
        while let Some(block) = state.worklist.pop_front() {
            self.analyze_block(opt, block, &mut state);
        }
    }

    fn analyze_block(&mut self, opt: &mut Optimizer, block: NodeIndex, state: &mut State) {
        let BlockSets { generated, kill } = block_sets(opt, block, state);

        let mut live_vars = generated.clone();

        for successor in opt.successors(block) {
            let successor = &self.live_vars[&successor];
            live_vars.extend(successor.difference(kill));
        }

        if live_vars != self.live_vars[&block] {
            state.worklist.extend(opt.predecessors(block));
            self.live_vars.insert(block, live_vars);
        }
    }
}

fn block_sets<'a>(opt: &mut Optimizer, block: NodeIndex, state: &'a mut State) -> &'a BlockSets {
    let block_sets = state.block_sets.entry(block);
    block_sets.or_insert_with(|| calculate_block_sets(opt, block))
}

fn calculate_block_sets(opt: &mut Optimizer, block: NodeIndex) -> BlockSets {
    let mut generated = HashSet::new();
    let mut kill = HashSet::new();

    let ops = opt.program[block].ops.clone();

    for op in ops.borrow_mut().values_mut().rev() {
        // Reads must be tracked after writes
        opt.visit_out(&mut op.out, |opt, var| {
            if let Some(id) = opt.local_variable_id(var) {
                kill.insert(id);
                generated.remove(&id);
            }
        });
        opt.visit_operation(&mut op.operation, &mut op.out, |opt, var| {
            if let Some(id) = opt.local_variable_id(var) {
                generated.insert(id);
            }
        });
    }

    BlockSets { generated, kill }
}
