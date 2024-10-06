use std::{
    collections::{HashMap, HashSet},
    mem::take,
};

use cubecl_core::ir::{Item, Operation, Variable};
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use crate::ControlFlow;

use super::Optimizer;

#[derive(Debug)]
pub struct SsaState<'a> {
    versions: HashMap<(u16, u8), u16>,
    visited_blocks: &'a mut HashSet<NodeIndex>,
    visited_edges: &'a mut HashSet<u32>,
    max_versions: &'a mut HashMap<(u16, u8), u16>,
}

#[derive(Debug, Clone)]
pub struct PhiEntry {
    pub block: NodeIndex,
    pub value: Variable,
}

#[derive(Debug, Clone)]
pub struct PhiInstruction {
    pub out: Variable,
    pub entries: Vec<PhiEntry>,
}

impl Optimizer {
    pub fn version_program(&mut self) {
        let versions: HashMap<_, _> = self.program.variables.keys().map(|key| (*key, 0)).collect();
        let mut visited_blocks = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut max_versions = versions.clone();
        let initial_state = SsaState {
            versions,
            visited_blocks: &mut visited_blocks,
            visited_edges: &mut visited_edges,
            max_versions: &mut max_versions,
        };
        self.version_block(self.entry(), initial_state);
    }

    fn version_block(&mut self, block: NodeIndex, mut state: SsaState<'_>) {
        self.version_block_ops(block, &mut state);

        let edges: Vec<_> = self
            .program
            .edges(block)
            .map(|it| (*it.weight(), it.target()))
            .collect();
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
                max_versions: state.max_versions,
            };

            if !edge_visited {
                self.version_phi(target, block, &new_state);
            }
            if !block_visited {
                self.version_block(target, new_state);
            }
        }
    }

    fn version_phi(&mut self, target: NodeIndex, source: NodeIndex, state: &SsaState<'_>) {
        let phi = self.program[target].phi_nodes.clone();
        for node in phi.borrow_mut().iter_mut() {
            let entry = node
                .entries
                .iter_mut()
                .find(|it| it.block == source)
                .unwrap();
            if let Some((id, depth, item, _)) = as_versioned(entry.value) {
                if self.program.variables.contains_key(&(id, depth)) {
                    let version = state.versions[&(id, depth)];
                    entry.value = Variable::Versioned {
                        id,
                        item,
                        depth,
                        version,
                    };
                }
            }
        }
    }

    fn version_block_ops(&mut self, block: NodeIndex, state: &mut SsaState<'_>) {
        for phi in self.program[block].phi_nodes.borrow_mut().iter_mut() {
            if let Some((id, depth, item, _)) = as_versioned(phi.out) {
                if self.program.variables.contains_key(&(id, depth)) {
                    let version = state.versions.get_mut(&(id, depth)).unwrap();
                    let max_version = state.max_versions.get_mut(&(id, depth)).unwrap();
                    *max_version += 1;
                    *version = *max_version;
                    phi.out = Variable::Versioned {
                        id,
                        item,
                        depth,
                        version: *version,
                    };
                }
            }
        }

        let mut ops = take(&mut *self.program[block].ops.borrow_mut());
        for operation in ops.values_mut() {
            self.version_reads(operation, state);
            self.version_writes(operation, state);
        }
        *self.program[block].ops.borrow_mut() = ops;
        match &mut *self.program[block].control_flow.borrow_mut() {
            super::ControlFlow::IfElse { cond, .. } => self.version_read(cond, state),
            ControlFlow::Switch { value, .. } => self.version_read(value, state),
            _ => {}
        }
    }

    fn version_reads(&mut self, op: &mut Operation, state: &mut SsaState<'_>) {
        self.visit_operation(op, |opt, var| opt.version_read(var, state), |_, _| {});
    }

    fn version_writes(&mut self, op: &mut Operation, state: &mut SsaState<'_>) {
        self.visit_operation(
            op,
            |_, _| {},
            |_, var| match var {
                Variable::Local { id, item, depth }
                | Variable::Versioned {
                    id, item, depth, ..
                } => {
                    if let Some(version) = state.versions.get_mut(&(*id, *depth)) {
                        let max_version = state.max_versions.get_mut(&(*id, *depth)).unwrap();
                        *max_version += 1;
                        *version = *max_version;
                        *var = Variable::Versioned {
                            id: *id,
                            item: *item,
                            depth: *depth,
                            version: *version,
                        }
                    }
                }
                _ => {}
            },
        );
    }

    fn version_read(&self, var: &mut Variable, state: &mut SsaState<'_>) {
        match var {
            Variable::Local { id, item, depth }
            | Variable::Versioned {
                id, item, depth, ..
            } => {
                if self.program.variables.contains_key(&(*id, *depth)) {
                    if let Some(version) = state.versions.get(&(*id, *depth)) {
                        *var = Variable::Versioned {
                            id: *id,
                            item: *item,
                            depth: *depth,
                            version: *version,
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn as_versioned(var: Variable) -> Option<(u16, u8, Item, u16)> {
    match var {
        Variable::Versioned {
            id,
            item,
            depth,
            version,
        } => Some((id, depth, item, version)),
        _ => None,
    }
}
