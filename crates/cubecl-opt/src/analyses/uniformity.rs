use alloc::vec::Vec;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_ir::{
    Builtin, Operation, OperationReflect, Operator, Plane, Synchronization, Value, ValueKind,
};
use petgraph::{graph::EdgeIndex, visit::EdgeRef};

use crate::{ControlFlow, Function, GlobalState, NodeIndex};

use super::Analysis;

#[derive(Default, Clone)]
pub struct Uniformity {
    block_uniformity: HashMap<NodeIndex, bool>,
    value_uniformity: HashMap<Value, bool>,
    visited: HashSet<EdgeIndex>,
}

impl Analysis for Uniformity {
    fn init(func: &mut Function, _: &GlobalState) -> Self {
        let mut this = Self::default();
        this.run(func);
        this
    }
}

impl Uniformity {
    fn run(&mut self, func: &Function) {
        let root = func.root;
        self.block_uniformity.insert(root, true);
        while self.analyze_block(func, root).is_none() {}
    }

    fn analyze_block(&mut self, func: &Function, block_id: NodeIndex) -> Option<()> {
        let block = func.block(block_id);
        let mut block_uniform = self.block_uniformity[&block_id];

        for phi in block.phi_nodes.borrow().iter() {
            let uniform = phi.entries.iter().all(|entry| {
                let block_uniform = self.is_block_uniform(entry.block);
                let value_uniform = self.is_val_uniform(entry.value);
                block_uniform && value_uniform
            }) && block_uniform;
            self.mark_uniformity(phi.out, uniform && block_uniform)?;
        }

        for inst in block.ops.borrow().values() {
            if inst.out.is_none() {
                continue;
            }
            let out = inst.out.unwrap();
            match &inst.operation {
                Operation::Plane(plane) => match plane {
                    // Elect returns true on only one unit, so it's always non-uniform
                    // Inclusive/exclusive scans are non-uniform by definition
                    Plane::Elect
                    | Plane::ExclusiveSum(_)
                    | Plane::InclusiveSum(_)
                    | Plane::ExclusiveProd(_)
                    | Plane::InclusiveProd(_) => self.mark_uniformity(out, false)?,
                    // Reductions are always uniform if executed in uniform control flow
                    Plane::Sum(_)
                    | Plane::Prod(_)
                    | Plane::Min(_)
                    | Plane::Max(_)
                    | Plane::All(_)
                    | Plane::Any(_)
                    | Plane::Ballot(_) => self.mark_uniformity(out, block_uniform)?,
                    // Broadcast maps to shuffle or broadcast, if id or value is uniform, so will
                    // the output, otherwise not.
                    Plane::Broadcast(op) => {
                        let input_uniform =
                            self.is_val_uniform(op.lhs) || self.is_val_uniform(op.rhs);
                        self.mark_uniformity(out, input_uniform && block_uniform)?;
                    }
                    // Shuffle operations: if offset/mask/delta is uniform, output is non-uniform
                    // (each thread gets a different value). If value is uniform, output is uniform.
                    Plane::Shuffle(op)
                    | Plane::ShuffleXor(op)
                    | Plane::ShuffleUp(op)
                    | Plane::ShuffleDown(op) => {
                        let input_uniform = self.is_val_uniform(op.lhs);
                        self.mark_uniformity(out, input_uniform && block_uniform)?;
                    }
                },
                Operation::Synchronization(sync) => match sync {
                    Synchronization::SyncCube | Synchronization::SyncStorage => {
                        block_uniform = true;
                    }
                    Synchronization::SyncAsyncProxyShared => {}
                    Synchronization::SyncPlane => {
                        // TODO: not sure
                    }
                },
                Operation::Operator(Operator::ReadBuiltin(builtin)) => {
                    self.mark_uniformity(out, is_builtin_uniform(builtin) && block_uniform)?;
                }
                Operation::Operator(Operator::ReadScalar(_)) => {
                    self.mark_uniformity(out, block_uniform);
                }
                op => {
                    let is_uniform =
                        op.is_pure() && self.is_all_uniform(op.args()) && block_uniform;
                    self.mark_uniformity(out, is_uniform)?;
                }
            }
        }

        match &*block.control_flow.borrow() {
            ControlFlow::IfElse {
                cond,
                then,
                or_else,
                merge,
            } => {
                let is_uniform = self.is_val_uniform(*cond);
                self.block_uniformity
                    .insert(*then, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*or_else, is_uniform && block_uniform);
                if let Some(merge) = merge {
                    self.block_uniformity.insert(*merge, block_uniform);
                }
            }
            ControlFlow::Switch {
                value,
                default,
                branches,
                merge,
            } => {
                let is_uniform = self.is_val_uniform(*value);
                self.block_uniformity
                    .insert(*default, is_uniform && block_uniform);
                for branch in branches {
                    self.block_uniformity
                        .insert(branch.1, is_uniform && block_uniform);
                }
                if let Some(merge) = merge {
                    self.block_uniformity.insert(*merge, block_uniform);
                }
            }
            ControlFlow::Loop {
                body,
                continue_target,
                merge,
            } => {
                // If we don't know the break condition, we can't detect whether it's uniform
                self.block_uniformity.insert(block_id, false);
                self.block_uniformity.insert(*body, false);
                self.block_uniformity.insert(*continue_target, false);
                self.block_uniformity.insert(*merge, false);
            }
            ControlFlow::LoopBreak {
                break_cond,
                body,
                continue_target,
                merge,
            } => {
                let is_uniform = self.is_val_uniform(*break_cond);
                self.block_uniformity
                    .insert(block_id, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*body, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*continue_target, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*merge, is_uniform && block_uniform);
            }
            ControlFlow::Return { .. } | ControlFlow::Unreachable => {}
            ControlFlow::None => {
                let successor = func.successors(block_id)[0];
                self.block_uniformity
                    .entry(successor)
                    .and_modify(|it| {
                        *it |= block_uniform;
                    })
                    .or_insert(block_uniform);
            }
        }

        for edge in func.edges(block_id) {
            if !self.visited.contains(&edge.id()) {
                self.visited.insert(edge.id());
                self.analyze_block(func, edge.target())?;
            }
        }

        Some(())
    }

    fn mark_uniformity(&mut self, val: Value, new_value: bool) -> Option<()> {
        if let Some(prev_value) = self.value_uniformity.get_mut(&val) {
            // If the value was already set before and has been invalidated, we need to revisit
            // all edges. This only happens for loopback edges, where an uninitialized value
            // was assumed to be uniform but actually isn't
            let invalidate = !new_value && *prev_value;
            *prev_value = *prev_value && new_value;
            if invalidate {
                self.visited.clear();
                return None;
            }
        } else {
            self.value_uniformity.insert(val, new_value);
        }
        Some(())
    }

    fn is_all_uniform(&self, args: Option<Vec<Value>>) -> bool {
        args.map(|it| it.iter().all(|it| self.is_val_uniform(*it)))
            .unwrap_or(false)
    }

    /// Whether a value is plane uniform
    pub fn is_val_uniform(&self, val: Value) -> bool {
        match val.kind {
            ValueKind::Constant(_) => true,
            ValueKind::Value { .. } => self.value_uniformity.get(&val).copied().unwrap_or(true),
        }
    }

    pub fn is_block_uniform(&self, block: NodeIndex) -> bool {
        self.block_uniformity.get(&block).copied().unwrap_or(true)
    }
}

fn is_builtin_uniform(builtin: &Builtin) -> bool {
    match builtin {
        Builtin::UnitPosPlane
        | Builtin::PlanePos
        | Builtin::AbsolutePos
        | Builtin::AbsolutePosX
        | Builtin::AbsolutePosY
        | Builtin::AbsolutePosZ
        | Builtin::UnitPos
        | Builtin::UnitPosX
        | Builtin::UnitPosY
        | Builtin::UnitPosZ => false,
        Builtin::CubePos
        | Builtin::CubePosX
        | Builtin::CubePosY
        | Builtin::CubePosZ
        | Builtin::CubePosCluster
        | Builtin::CubePosClusterX
        | Builtin::CubePosClusterY
        | Builtin::CubePosClusterZ
        | Builtin::CubeDim
        | Builtin::CubeDimX
        | Builtin::CubeDimY
        | Builtin::CubeDimZ
        | Builtin::CubeClusterDim
        | Builtin::CubeClusterDimX
        | Builtin::CubeClusterDimY
        | Builtin::CubeClusterDimZ
        | Builtin::CubeCount
        | Builtin::CubeCountX
        | Builtin::CubeCountY
        | Builtin::CubeCountZ
        | Builtin::PlaneDim => true,
    }
}
