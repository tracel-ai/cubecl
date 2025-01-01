use std::ops::Deref;

use crate::NodeIndex;
use petgraph::algo::dominators;

use super::Analysis;

pub struct Dominators(dominators::Dominators<NodeIndex>);
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
    fn init(opt: &mut crate::Optimizer) -> Self {
        Dominators(dominators::simple_fast(&opt.program.graph, opt.entry()))
    }
}

impl Analysis for PostDominators {
    fn init(opt: &mut crate::Optimizer) -> Self {
        let mut reversed = opt.program.graph.clone();
        reversed.reverse();
        PostDominators(dominators::simple_fast(&reversed, opt.ret))
    }
}
