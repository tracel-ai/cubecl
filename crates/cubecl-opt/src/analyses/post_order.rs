use crate::NodeIndex;
use petgraph::visit::{DfsPostOrder, Walker};

use super::Analysis;

pub struct PostOrder(Vec<NodeIndex>);

impl Analysis for PostOrder {
    fn init(opt: &mut crate::Optimizer) -> Self {
        let po = DfsPostOrder::new(&opt.program.graph, opt.entry());
        PostOrder(po.iter(&opt.program.graph).collect())
    }
}

impl PostOrder {
    pub fn forward(&self) -> Vec<NodeIndex> {
        self.0.clone()
    }

    pub fn reverse(&self) -> Vec<NodeIndex> {
        self.0.clone().into_iter().rev().collect()
    }
}
