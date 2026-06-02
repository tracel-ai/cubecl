use core::mem::take;

use alloc::vec::Vec;
use cubecl_ir::{Id, Instruction, Type, Variable, VariableKind};
use hashbrown::{HashMap, HashSet};
use petgraph::visit::EdgeRef;

use crate::{ControlFlow, EdgeIndex, Function, GlobalState, NodeIndex};

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
    pub value: Variable,
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
    pub out: Variable,
    /// The set of `block`-`value` pairs for the phi instruction
    pub entries: Vec<PhiEntry>,
}

impl Function {
    /// Version all variables in the program so they are each assigned to exactly once.
    pub(crate) fn version_program(&mut self, global_state: &GlobalState) {
        let versions: HashMap<_, _> = self.variables.keys().map(|key| (*key, 0)).collect();
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
                && self.variables.contains_key(&id)
            {
                let id = state.versions[&id];
                entry.value = Variable::new(VariableKind::LocalConst { id }, item);
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
                && self.variables.contains_key(&id)
            {
                let version = state.versions.get_mut(&id).unwrap();
                let out = global_state.root_scope.create_local(item);
                *version = out.index().unwrap();
                phi.out = out;
            }
        }

        let mut ops = take(&mut *self[block].ops.borrow_mut());
        for operation in ops.values_mut() {
            self.version_reads(global_state, operation, state);
            self.version_writes(operation, state, global_state);
        }
        *self[block].ops.borrow_mut() = ops;
        match &mut *self[block].control_flow.borrow_mut() {
            ControlFlow::IfElse { cond, .. } => self.version_read(cond, state),
            ControlFlow::LoopBreak { break_cond, .. } => self.version_read(break_cond, state),
            ControlFlow::Switch { value, .. } => self.version_read(value, state),
            ControlFlow::Loop { .. } => {}
            ControlFlow::Return { value } => {
                if let Some(value) = value {
                    self.version_read(value, state);
                }
            }
            ControlFlow::Unreachable | ControlFlow::None => {}
        }
    }

    fn version_reads(
        &mut self,
        global_state: &GlobalState,
        op: &mut Instruction,
        state: &mut SsaState<'_>,
    ) {
        self.visit_operation(global_state, &mut op.operation, |func, var| {
            func.version_read(var, state)
        });
    }

    fn version_writes(
        &mut self,
        op: &mut Instruction,
        state: &mut SsaState<'_>,
        global_state: &GlobalState,
    ) {
        self.visit_out(&mut op.out, |_, var| {
            if let VariableKind::LocalMut { id } = var.kind
                && let Some(version) = state.versions.get_mut(&id)
            {
                let new_var = global_state.root_scope.create_local(var.ty);
                *version = new_var.index().unwrap();
                *var = new_var;
            }
        });
    }

    fn version_read(&self, var: &mut Variable, state: &mut SsaState<'_>) {
        if let VariableKind::LocalMut { id } = var.kind
            && self.variables.contains_key(&id)
            && let Some(id) = state.versions.get(&id)
        {
            *var = Variable::new(VariableKind::LocalConst { id: *id }, var.ty)
        }
    }
}

fn as_local(var: Variable) -> Option<(Id, Type)> {
    match var.kind {
        VariableKind::LocalConst { id } => Some((id, var.ty)),
        _ => None,
    }
}
