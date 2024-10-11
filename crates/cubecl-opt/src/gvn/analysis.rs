use std::{
    cell::RefCell,
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use derive_more::derive::{Deref, DerefMut};
use hashlink::LinkedHashSet;
use petgraph::{
    algo::dominators::{self, Dominators},
    graph::NodeIndex,
};

use crate::{AtomicCounter, Optimizer};

use super::{as_var, Expr, GlobalNumberTable, Local};

#[derive(Clone)]
pub struct BlockNumbers {
    pub avail_in: GlobalNumberTable,
    pub avail_out: GlobalNumberTable,
    pub anticip_in: GlobalNumberTable,
    pub anticip_out: GlobalNumberTable,
}

impl BlockNumbers {
    pub fn new(class_id: AtomicCounter, globals: Rc<RefCell<HashMap<Expr, usize>>>) -> Self {
        Self {
            avail_in: GlobalNumberTable::new(class_id.clone(), globals.clone()),
            avail_out: GlobalNumberTable::new(class_id.clone(), globals.clone()),
            anticip_in: GlobalNumberTable::new(class_id.clone(), globals.clone()),
            anticip_out: GlobalNumberTable::new(class_id.clone(), globals.clone()),
        }
    }
}

#[derive(Deref, DerefMut)]
pub struct GlobalNumberGraph {
    #[deref]
    #[deref_mut]
    nodes: HashMap<NodeIndex, BlockNumbers>,
    class_id: AtomicCounter,
    pub(crate) globals: Rc<RefCell<HashMap<Expr, usize>>>,
    table: GlobalNumberTable,
    block_sets: HashMap<NodeIndex, BlockSets>,
    dominators: Dominators<NodeIndex>,
    post_dominators: Dominators<NodeIndex>,
}

struct BlockSets {
    exp_gen: HashMap<usize, Expr>,
    phi_gen: HashSet<Local>,
    tmp_gen: HashMap<usize, Local>,
}

impl GlobalNumberGraph {
    pub fn new(opt: &Optimizer) -> GlobalNumberGraph {
        let doms = dominators::simple_fast(&opt.program.graph, opt.entry());
        let mut rev_graph = opt.program.graph.clone();
        rev_graph.reverse();
        let post_doms = dominators::simple_fast(&rev_graph, opt.ret);
        let class_id = AtomicCounter::default();
        GlobalNumberGraph {
            nodes: Default::default(),
            class_id: class_id.clone(),
            globals: Default::default(),
            table: GlobalNumberTable::new(class_id, Default::default()),
            block_sets: Default::default(),
            dominators: doms,
            post_dominators: post_doms,
        }
    }

    pub fn build_sets(&mut self, opt: &Optimizer) {
        self.build_block_sets_fwd(opt, self.dominators.root(), HashMap::new());
    }

    fn build_block_sets_fwd(
        &mut self,
        opt: &Optimizer,
        block: NodeIndex,
        mut leaders: HashMap<usize, Expr>,
    ) {
        let mut exp_gen = leaders.clone();
        let mut phi_gen = HashSet::new();
        let mut tmp_gen = HashMap::new();

        for phi in opt.program[block].phi_nodes.borrow().iter() {
            let value = as_var(&phi.out);
            phi_gen.insert(value);
        }

        for op in opt.program[block].ops.borrow().values() {
            let instruction = self.table.class_of_operation(op);
            if let Some(instruction) = instruction {
                if let Entry::Vacant(e) = exp_gen.entry(instruction.class) {
                    e.insert(instruction.rhs);
                    tmp_gen.insert(instruction.class, instruction.out);
                }
            }
        }

        leaders.extend(tmp_gen.iter().map(|it| (*it.0, Expr::Variable(*it.1))));

        let sets = BlockSets {
            exp_gen,
            phi_gen,
            tmp_gen,
        };
        self.block_sets.insert(block, sets);
        let successors: Vec<_> = self.dominators.immediately_dominated_by(block).collect();
        for successor in successors {
            self.build_block_sets_fwd(opt, successor, leaders.clone());
        }
    }

    fn build_block_sets_bckwd(&mut self, opt: &Optimizer, block: NodeIndex) -> bool {
        let mut changed = false;

        changed
    }
}
