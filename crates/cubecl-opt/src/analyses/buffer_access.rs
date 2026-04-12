use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use cubecl_ir::Id;

use crate::{Function, GlobalState, NodeIndex, local_variable_id};

use super::Analysis;

pub struct BufferAccessibility {
    readable: bool,
    writable: bool,
}

#[derive(Debug)]
pub struct BufferAccesses {
    pub buffers: Vec<BufferAccessibility>,
}

impl Deref for BufferAccesses {
    type Target = Vec<BufferAccessibility>;

    fn deref(&self) -> &Self::Target {
        &self.buffers
    }
}

impl BufferAccesses {
    pub fn new(opt: &mut Function) -> Self {
        let nodes = opt.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut writes: HashMap<NodeIndex, HashSet<Id>> = nodes.collect();
        for block in opt.node_ids() {
            let ops = opt[block].ops.clone();
            for inst in ops.borrow().values() {
                if let Some(id) = inst.out.as_ref().and_then(local_variable_id) {
                    writes.get_mut(&block).unwrap().insert(id);
                }
            }
        }
        Writes { writes }
    }
}

impl Analysis for Writes {
    fn init(opt: &mut crate::Function, _: &GlobalState) -> Self {
        Writes::new(opt)
    }
}
