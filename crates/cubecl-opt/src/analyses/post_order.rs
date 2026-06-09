use crate::{GlobalState, NodeIndex};
use alloc::vec::Vec;
use petgraph::visit::{DfsPostOrder, Walker};

use super::Analysis;

pub struct PostOrder(Vec<NodeIndex>);

impl Analysis for PostOrder {
    fn init(func: &mut crate::Function, _: &GlobalState) -> Self {
        let po = DfsPostOrder::new(&func.graph, func.root);
        PostOrder(po.iter(&func.graph).collect())
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
