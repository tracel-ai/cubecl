use alloc::{collections::VecDeque, vec::Vec};
use core::ops::Deref;

use crate::{Function, GlobalState, NodeIndex};
use hashbrown::{HashMap, HashSet};
use petgraph::algo::dominators;

use super::Analysis;

/// Dominator tree for the program graph
pub struct Dominators(dominators::Dominators<NodeIndex>);
/// Post dominator tree for the program graph
pub struct PostDominators(dominators::Dominators<NodeIndex>);

impl Deref for Dominators {
    type Target = dominators::Dominators<NodeIndex>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for PostDominators {
    type Target = dominators::Dominators<NodeIndex>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Analysis for Dominators {
    fn init(func: &mut crate::Function, _: &GlobalState) -> Self {
        Dominators(dominators::simple_fast(&func.graph, func.root))
    }
}

impl Analysis for PostDominators {
    fn init(func: &mut crate::Function, _: &GlobalState) -> Self {
        let mut reversed = func.graph.clone();
        reversed.reverse();
        PostDominators(dominators::simple_fast(&reversed, func.ret))
    }
}

impl Dominators {
    pub fn breadth_first_nodes(&self) -> Vec<NodeIndex> {
        let mut out = Vec::new();
        let mut worklist = VecDeque::new();
        worklist.push_back(self.root());
        while let Some(node) = worklist.pop_front() {
            out.push(node);
            worklist.extend(self.immediately_dominated_by(node));
        }
        out
    }
}

/// Dominance frontiers for each block
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
    pub fn new(func: &mut Function, state: &GlobalState) -> Self {
        let doms = func.analysis::<Dominators>(state);
        let nodes = func.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut dom_frontiers: HashMap<NodeIndex, HashSet<NodeIndex>> = nodes.collect();

        for node in func.node_ids() {
            let predecessors = func.predecessors(node);
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
    fn init(func: &mut Function, state: &GlobalState) -> Self {
        DomFrontiers::new(func, state)
    }
}
