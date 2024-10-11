use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
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

use super::{GlobalNumberTable, GlobalValue};

#[derive(Clone)]
pub struct BlockNumbers {
    pub avail_in: GlobalNumberTable,
    pub avail_out: GlobalNumberTable,
    pub anticip_in: GlobalNumberTable,
    pub anticip_out: GlobalNumberTable,
}

impl BlockNumbers {
    pub fn new(class_id: AtomicCounter, globals: Rc<RefCell<HashMap<GlobalValue, usize>>>) -> Self {
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
    pub(crate) globals: Rc<RefCell<HashMap<GlobalValue, usize>>>,
    dominators: Dominators<NodeIndex>,
}

impl GlobalNumberGraph {
    pub fn new(opt: &Optimizer) -> GlobalNumberGraph {
        let doms = dominators::simple_fast(&opt.program.graph, opt.entry());
        GlobalNumberGraph {
            nodes: Default::default(),
            class_id: Default::default(),
            globals: Default::default(),
            dominators: doms,
        }
    }

    pub fn build_sets(&mut self, opt: &Optimizer) {
        self.built_block_sets(opt, self.dominators.root());
    }

    fn built_block_sets(&mut self, opt: &Optimizer, block: NodeIndex) {
        let exp_gen = HashSet::new();
        let phi_gen = HashSet::new();
        let tmp_gen = HashSet::new();
    }
}
