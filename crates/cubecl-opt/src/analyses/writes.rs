use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use cubecl_ir::{Id, Memory, Operation, Operator};

use crate::{
    Function, GlobalState, NodeIndex, analyses::pointer_source::PointerSource, local_variable_id,
};

use super::Analysis;

#[derive(Debug)]
pub struct Writes {
    /// The variables written to by each block.
    writes: HashMap<NodeIndex, HashSet<Id>>,
}

impl Deref for Writes {
    type Target = HashMap<NodeIndex, HashSet<Id>>;

    fn deref(&self) -> &Self::Target {
        &self.writes
    }
}

impl Writes {
    pub fn new(opt: &mut Function, state: &GlobalState) -> Self {
        let ptr_source = opt.analysis::<PointerSource>(state);
        let nodes = opt.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut writes: HashMap<NodeIndex, HashSet<Id>> = nodes.collect();
        for block in opt.node_ids() {
            let ops = opt[block].ops.clone();
            for inst in ops.borrow().values() {
                if let Some(id) = inst.out.as_ref().and_then(local_variable_id) {
                    writes.get_mut(&block).unwrap().insert(id);
                }
                if let Operation::Memory(Memory::Store(var)) = &inst.operation
                    && let Some(source) = ptr_source.get(&var.ptr)
                    && let Some(id) = local_variable_id(&source)
                {
                    writes.get_mut(&block).unwrap().insert(id);
                }
                if let Operation::Operator(Operator::InsertComponent(_)) = &inst.operation
                    && let Some(source) = ptr_source.get(&inst.out())
                    && let Some(id) = local_variable_id(&source)
                {
                    writes.get_mut(&block).unwrap().insert(id);
                }
            }
        }
        Writes { writes }
    }
}

impl Analysis for Writes {
    fn init(opt: &mut crate::Function, state: &GlobalState) -> Self {
        Writes::new(opt, state)
    }
}
