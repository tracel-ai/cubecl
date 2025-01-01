use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use crate::{NodeIndex, Optimizer};

use super::Analysis;

pub struct Writes {
    /// The variables written to by each block.
    writes: HashMap<NodeIndex, HashSet<(u16, u8)>>,
}

impl Deref for Writes {
    type Target = HashMap<NodeIndex, HashSet<(u16, u8)>>;

    fn deref(&self) -> &Self::Target {
        &self.writes
    }
}

impl Writes {
    pub fn new(opt: &mut Optimizer) -> Self {
        let nodes = opt.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut writes: HashMap<NodeIndex, HashSet<(u16, u8)>> = nodes.collect();
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for inst in ops.borrow().values() {
                if let Some(id) = inst.out.as_ref().and_then(|it| opt.local_variable_id(it)) {
                    writes.get_mut(&block).unwrap().insert(id);
                }
            }
        }
        Writes { writes }
    }
}

impl Analysis for Writes {
    fn init(opt: &mut crate::Optimizer) -> Self {
        Writes::new(opt)
    }
}
