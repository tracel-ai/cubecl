use core::mem::take;

use alloc::vec::Vec;
use cubecl_ir::{Id, Instruction, Memory, Operation, Type, Value, ValueKind};
use hashbrown::{HashMap, HashSet};
use petgraph::visit::EdgeRef;

use crate::{EdgeIndex, Function, GlobalState, NodeIndex};

/// The state required by the SSA transform
#[derive(Debug)]
pub struct SsaState<'a> {
    versions: HashMap<Id, Id>,
    visited_blocks: &'a mut HashSet<NodeIndex>,
    visited_edges: &'a mut HashSet<EdgeIndex>,
}

/// An entry in the phi instruction. Contains the variable ID that should be used when coming from
/// `block`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhiEntry {
    pub block: NodeIndex,
    pub value: Value,
}

/// A phi node that picks its value based on the `BasicBlock` that came immediately before.
/// For more information, see <https://en.wikipedia.org/wiki/Static_single-assignment_form>
///
/// # Example
/// ```ignore
/// if cond {
///     result = "heads";
/// } else {
///     result = "tails";
/// }
/// ```
/// would translate to the following SSA graph:
/// ```ignore
/// bb1: {
///     branch if cond { bb2 } else { bb3 };
/// }
///
/// bb2: {
///     let result.v1 = "heads";
///     branch bb4;
/// }
///
/// bb3: {
///     let result.v2 = "tails";
///     branch bb4;
/// }
///
/// bb4: {
///     let result.v3 = phi [bb2: result.v1] [bb3: result.v2];
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PhiInstruction {
    /// The out variable for the phi instruction
    pub out: Value,
    /// The set of `block`-`value` pairs for the phi instruction
    pub entries: Vec<PhiEntry>,
}

impl Function {
    /// Version all variables in the program so they are each assigned to exactly once.
    pub(crate) fn version_program(&mut self, global_state: &GlobalState) {
        let locals = self.destructurable_local_memories();
        let versions: HashMap<_, _> = locals.keys().map(|key| (*key, 0)).collect();
        let mut visited_blocks = HashSet::new();
        let mut visited_edges = HashSet::new();
        let initial_state = SsaState {
            versions,
            visited_blocks: &mut visited_blocks,
            visited_edges: &mut visited_edges,
        };
        self.version_block(global_state, self.root, initial_state);
    }

    fn version_block(
        &mut self,
        global_state: &GlobalState,
        block: NodeIndex,
        mut state: SsaState<'_>,
    ) {
        self.version_block_ops(global_state, block, &mut state);

        let edges: Vec<_> = self.edges(block).map(|it| (it.id(), it.target())).collect();
        let state = &mut state;
        for (edge_id, target) in edges {
            let edge_visited = state.visited_edges.contains(&edge_id);
            state.visited_edges.insert(edge_id);
            let block_visited = state.visited_blocks.contains(&target);
            state.visited_blocks.insert(block);

            let new_state = SsaState {
                versions: state.versions.clone(),
                visited_blocks: state.visited_blocks,
                visited_edges: state.visited_edges,
            };

            if !edge_visited {
                self.version_phi(target, block, &new_state);
            }
            if !block_visited {
                self.version_block(global_state, target, new_state);
            }
        }
    }

    /// Version the phi entry for this edge
    fn version_phi(&mut self, target: NodeIndex, source: NodeIndex, state: &SsaState<'_>) {
        let phi = self[target].phi_nodes.clone();
        for node in phi.borrow_mut().iter_mut() {
            let entry = node
                .entries
                .iter_mut()
                .find(|it| it.block == source)
                .unwrap();
            if let Some((id, item)) = as_local(entry.value)
                && self.destructurable_local_memories().contains_key(&id)
            {
                let id = state.versions[&id];
                entry.value = Value::new(id, item);
            }
        }
    }

    /// Version the operations for this block
    fn version_block_ops(
        &mut self,
        global_state: &GlobalState,
        block: NodeIndex,
        state: &mut SsaState<'_>,
    ) {
        for phi in self[block].phi_nodes.borrow_mut().iter_mut() {
            if let Some((id, item)) = as_local(phi.out)
                && self.destructurable_local_memories().contains_key(&id)
            {
                let version = state.versions.get_mut(&id).unwrap();
                let out = global_state.root_scope.create_value(item);
                *version = out.id();
                phi.out = out;
            }
        }

        let ops = take(&mut *self[block].ops.borrow_mut());
        let ops = ops.into_iter().flat_map(|(_, mut instruction)| {
            self.version_loads(&mut instruction, state);
            self.version_stores(&mut instruction, state, global_state);
            if let Operation::DeclareVariable { .. } = &instruction.operation
                && state.versions.contains_key(&instruction.out().id())
            {
                None
            } else {
                Some(instruction)
            }
        });
        *self[block].ops.borrow_mut() = ops.collect();
    }

    fn version_loads(&mut self, inst: &mut Instruction, state: &mut SsaState<'_>) {
        if let Operation::Memory(Memory::Load(ptr)) = inst.operation
            && let Some(id) = state.versions.get(&ptr.id())
        {
            let new_val = Value::new(*id, inst.out().ty);
            *inst = Instruction::new(Operation::Copy(new_val), inst.out())
        }
    }

    fn version_stores(
        &mut self,
        inst: &mut Instruction,
        state: &mut SsaState<'_>,
        global_state: &GlobalState,
    ) {
        if let Operation::Memory(Memory::Store(store)) = &mut inst.operation
            && let Some(version) = state.versions.get_mut(&store.ptr.id())
        {
            let new_val = global_state.root_scope.create_value(store.value.ty);
            *version = new_val.id();
            *inst = Instruction::new(Operation::Copy(store.value), new_val);
        }
    }
}

fn as_local(var: Value) -> Option<(Id, Type)> {
    match var.kind {
        ValueKind::Value { id } => Some((id, var.ty)),
        _ => None,
    }
}
