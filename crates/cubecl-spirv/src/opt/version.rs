use std::collections::{HashMap, HashSet};

use cubecl_core::ir::{Item, Operation, Variable};
use petgraph::{graph::NodeIndex, visit::EdgeRef};
use serde::{Deserialize, Serialize};

use super::Optimizer;

#[derive(Debug)]
pub struct SsaState<'a> {
    versions: HashMap<(u16, u8), u16>,
    visited_blocks: &'a mut HashSet<NodeIndex>,
    visited_edges: &'a mut HashSet<u32>,
    max_versions: &'a mut HashMap<(u16, u8), u16>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct PhiEntry {
    pub block: NodeIndex,
    pub value: (u16, u8, u16),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PhiInstruction {
    pub out: (u16, u8, u16),
    pub entries: Vec<PhiEntry>,
    pub item: Item,
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
        let mut phi = self.program[target].phi_nodes.drain(..).collect::<Vec<_>>();
        for node in &mut phi {
            let entry = node
                .entries
                .iter_mut()
                .find(|it| it.block == source)
                .unwrap();
            let id = (entry.value.0, entry.value.1);
            let version = state.versions[&id];
            entry.value = (id.0, id.1, version);
        }
        self.program[target].phi_nodes = phi;
    }

    fn version_block_ops(&mut self, block: NodeIndex, state: &mut SsaState<'_>) {
        for phi in &mut self.program[block].phi_nodes {
            let (id, depth, _) = phi.out;
            let version = state.versions.get_mut(&(id, depth)).unwrap();
            let max_version = state.max_versions.get_mut(&(id, depth)).unwrap();
            *max_version += 1;
            *version = *max_version;
            phi.out = (id, depth, *version)
        }

        let mut ops = self.program[block].ops.drain(..).collect::<Vec<_>>();
        for operation in &mut ops {
            self.version_reads(operation, state);
            self.version_writes(operation, state);
        }
        self.program[block].ops = ops;
        match &mut self.program[block].control_flow {
            super::ControlFlow::If { cond, .. } | super::ControlFlow::IfElse { cond, .. } => {
                version_read(cond, state)
            }
            super::ControlFlow::Switch { value, .. } => version_read(value, state),
            _ => {}
        }
    }

    fn version_reads(&mut self, op: &mut Operation, state: &mut SsaState<'_>) {
        self.visit_operation(op, |_, var| version_read(var, state), |_, _| {});
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
}

fn version_read(var: &mut Variable, state: &mut SsaState<'_>) {
    match var {
        Variable::Local { id, item, depth }
        | Variable::Versioned {
            id, item, depth, ..
        } => {
            if let Some(version) = state.versions.get(&(*id, *depth)) {
                *var = Variable::Versioned {
                    id: *id,
                    item: *item,
                    depth: *depth,
                    version: *version,
                }
            }
        }
        _ => {}
    }
}
