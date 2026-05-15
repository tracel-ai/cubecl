use core::ops::Deref;

use cubecl_ir::Id;
use hashbrown::{HashMap, HashSet};

use crate::{Function, GlobalState, NodeIndex, local_variable_id};

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
    pub fn new(func: &mut Function, state: &GlobalState) -> Self {
        let nodes = func.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut writes: HashMap<NodeIndex, HashSet<Id>> = nodes.collect();
        for block in func.node_ids() {
            let ops = func[block].ops.clone();
            for inst in ops.borrow_mut().values_mut() {
                func.visit_instruction_write(state, inst, |_, var| {
                    if let Some(id) = local_variable_id(var) {
                        writes.get_mut(&block).unwrap().insert(id);
                    }
                });
            }
        }
        Writes { writes }
    }
}

impl Analysis for Writes {
    fn init(func: &mut crate::Function, state: &GlobalState) -> Self {
        Writes::new(func, state)
    }
}
