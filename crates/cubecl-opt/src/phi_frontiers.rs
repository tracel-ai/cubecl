use alloc::vec::Vec;
use cubecl_environment::collections::HashMap;
use cubecl_ir::{AddressSpace, Id, Type, Value};
use petgraph::graph::NodeIndex;

use crate::{
    Function, GlobalState, MemoryBlock,
    analyses::{dominance::DomFrontiers, liveness::Liveness, writes::LocalStores},
};

use super::version::{PhiEntry, PhiInstruction};

impl Function {
    /// Places a phi node for each live variable at each frontier
    pub fn place_phi_nodes(&mut self, state: &GlobalState) {
        let locals = self.destructurable_local_memories();
        let writes = self.analysis::<LocalStores>(state);
        let liveness = self.analysis::<Liveness>(state);
        let dom_frontiers = self.analysis::<DomFrontiers>(state);

        for (local_id, mem) in locals {
            let mut workset: Vec<_> = self
                .node_ids()
                .iter()
                .filter(|index| writes[*index].contains(&local_id))
                .copied()
                .collect();
            let mut considered = workset.clone();
            let mut already_inserted = Vec::new();

            while let Some(node) = workset.pop() {
                for frontier in dom_frontiers[&node].clone() {
                    if already_inserted.contains(&frontier) || liveness.is_dead(frontier, local_id)
                    {
                        continue;
                    }
                    self.insert_phi(frontier, local_id, mem.value_ty);
                    already_inserted.push(frontier);
                    if !considered.contains(&frontier) {
                        workset.push(frontier);
                        considered.push(frontier);
                    }
                }
            }
        }
    }

    /// Insert a phi node for variable `id` at `block`
    pub fn insert_phi(&mut self, block: NodeIndex, id: Id, item: Type) {
        let val = Value::new(id, item);
        let entries = self.predecessors(block).into_iter().map(|pred| PhiEntry {
            block: pred,
            value: val,
        });
        let phi = PhiInstruction {
            out: val,
            entries: entries.collect(),
        };
        self[block].phi_nodes.borrow_mut().push(phi);
    }

    /// Returns all pointers to local stack space that are [destructurable](Type::is_destructurable)
    pub fn destructurable_local_memories(&self) -> HashMap<Id, MemoryBlock> {
        let locals = self.memories.iter().filter(|(_, mem)| {
            matches!(mem.address_space, AddressSpace::Local) && mem.value_ty.is_destructurable()
        });
        locals.map(|(k, v)| (*k, *v)).collect()
    }
}
