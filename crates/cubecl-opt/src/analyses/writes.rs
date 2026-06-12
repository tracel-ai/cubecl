use core::ops::Deref;

use cubecl_ir::{Id, Memory, Operation};
use hashbrown::{HashMap, HashSet};

use crate::{Function, GlobalState, NodeIndex};

use super::Analysis;

#[derive(Debug)]
pub struct LocalStores {
    /// The memories stored to by each block.
    stores: HashMap<NodeIndex, HashSet<Id>>,
}

impl Deref for LocalStores {
    type Target = HashMap<NodeIndex, HashSet<Id>>;

    fn deref(&self) -> &Self::Target {
        &self.stores
    }
}

impl LocalStores {
    pub fn new(func: &mut Function, _state: &GlobalState) -> Self {
        let nodes = func.node_ids().into_iter().map(|it| (it, HashSet::new()));
        let mut stores: HashMap<NodeIndex, HashSet<Id>> = nodes.collect();
        let locals = func.destructurable_local_memories();

        for block in func.node_ids() {
            let ops = func[block].ops.borrow();
            for inst in ops.values() {
                if let Operation::Memory(Memory::Store(store)) = &inst.operation
                    && locals.contains_key(&store.ptr.id())
                {
                    stores.get_mut(&block).unwrap().insert(store.ptr.id());
                }
            }
        }
        LocalStores { stores }
    }
}

impl Analysis for LocalStores {
    fn init(func: &mut crate::Function, state: &GlobalState) -> Self {
        LocalStores::new(func, state)
    }
}
