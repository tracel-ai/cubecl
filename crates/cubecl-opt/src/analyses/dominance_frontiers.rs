use crate::{NodeIndex, Optimizer};
use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use super::{dominators::Dominators, Analysis};

pub struct DomFrontiers {
    /// The dominance frontiers of each block (where phi nodes must be inserted).
    dom_frontiers: HashMap<NodeIndex, HashSet<NodeIndex>>,
}

impl Deref for DomFrontiers {
    type Target = HashMap<NodeIndex, HashSet<NodeIndex>>;

    fn deref(&self) -> &Self::Target {
        &self.dom_frontiers
    }
}

impl DomFrontiers {
    /// Find dominance frontiers for each block
    pub fn new(opt: &mut Optimizer) -> Self {
        let doms = opt.analysis::<Dominators>();
        let nodes = opt.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut dom_frontiers: HashMap<NodeIndex, HashSet<NodeIndex>> = nodes.collect();

        for node in opt.node_ids() {
            let predecessors = opt.predecessors(node);
            if predecessors.len() >= 2 {
                for predecessor in predecessors {
                    let mut runner = predecessor;
                    while runner != doms.immediate_dominator(node).unwrap() {
                        dom_frontiers.get_mut(&runner).unwrap().insert(node);
                        runner = doms.immediate_dominator(runner).unwrap();
                    }
                }
            }
        }
        Self { dom_frontiers }
    }
}

impl Analysis for DomFrontiers {
    fn init(opt: &mut Optimizer) -> Self {
        DomFrontiers::new(opt)
    }
}
