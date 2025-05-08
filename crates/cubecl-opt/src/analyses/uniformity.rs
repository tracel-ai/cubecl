use cubecl_ir::{
    Builtin, Operation, OperationReflect, Plane, Synchronization, Variable, VariableKind,
};
use petgraph::{graph::EdgeIndex, visit::EdgeRef};
use std::collections::{HashMap, HashSet};

use crate::{ControlFlow, NodeIndex, Optimizer};

use super::Analysis;

#[derive(Default, Clone)]
pub struct Uniformity {
    block_uniformity: HashMap<NodeIndex, bool>,
    variable_uniformity: HashMap<Variable, bool>,
    visited: HashSet<EdgeIndex>,
}

impl Analysis for Uniformity {
    fn init(opt: &mut Optimizer) -> Self {
        let mut this = Self::default();
        this.run(opt);
        this
    }
}

impl Uniformity {
    fn run(&mut self, opt: &Optimizer) {
        let root = opt.entry();
        self.block_uniformity.insert(root, true);
        while self.analyze_block(opt, root).is_none() {}
    }

    fn analyze_block(&mut self, opt: &Optimizer, block_id: NodeIndex) -> Option<()> {
        let block = opt.block(block_id);
        let mut block_uniform = self.block_uniformity[&block_id];

        for phi in block.phi_nodes.borrow().iter() {
            let uniform = phi.entries.iter().all(|entry| {
                let block_uniform = self.is_block_uniform(entry.block);
                let value_uniform = self.is_var_uniform(entry.value);
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
                            self.is_var_uniform(op.lhs) || self.is_var_uniform(op.rhs);
                        self.mark_uniformity(out, input_uniform && block_uniform)?;
                    }
                },
                Operation::Synchronization(sync) => match sync {
                    Synchronization::SyncCube | Synchronization::SyncStorage => {
                        block_uniform = true;
                    }
                    Synchronization::SyncProxyShared => {}
                    Synchronization::SyncPlane => {
                        // TODO: not sure
                    }
                },
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
                let is_uniform = self.is_var_uniform(*cond);
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
                let is_uniform = self.is_var_uniform(*value);
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
                let is_uniform = self.is_var_uniform(*break_cond);
                self.block_uniformity
                    .insert(block_id, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*body, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*continue_target, is_uniform && block_uniform);
                self.block_uniformity
                    .insert(*merge, is_uniform && block_uniform);
            }
            ControlFlow::Return => {}
            ControlFlow::None => {
                let successor = opt.successors(block_id)[0];
                self.block_uniformity
                    .entry(successor)
                    .and_modify(|it| {
                        *it |= block_uniform;
                    })
                    .or_insert(block_uniform);
            }
        }

        for edge in opt.program.edges(block_id) {
            if !self.visited.contains(&edge.id()) {
                self.visited.insert(edge.id());
                self.analyze_block(opt, edge.target())?;
            }
        }

        Some(())
    }

    fn mark_uniformity(&mut self, var: Variable, value: bool) -> Option<()> {
        if let Some(val) = self.variable_uniformity.get_mut(&var) {
            // If the value was already set before and has been invalidated, we need to revisit
            // all edges. This only happens for loopback edges, where an uninitialized variable
            // was assumed to be uniform but actually isn't
            let invalidate = !value && *val;
            *val = *val && value;
            if invalidate {
                self.visited.clear();
                return None;
            }
        } else {
            self.variable_uniformity.insert(var, value);
        }
        Some(())
    }

    fn is_all_uniform(&self, args: Option<Vec<Variable>>) -> bool {
        args.map(|it| it.iter().all(|it| self.is_var_uniform(*it)))
            .unwrap_or(false)
    }

    /// Whether a variable is plane uniform
    pub fn is_var_uniform(&self, var: Variable) -> bool {
        match var.kind {
            VariableKind::ConstantArray { .. }
            | VariableKind::SharedMemory { .. }
            | VariableKind::GlobalInputArray(_)
            | VariableKind::GlobalOutputArray(_)
            | VariableKind::GlobalScalar(_)
            | VariableKind::ConstantScalar(_) => true,
            VariableKind::Builtin(builtin) => match builtin {
                Builtin::UnitPosPlane
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
            },
            VariableKind::LocalMut { .. } => false,
            VariableKind::LocalArray { .. }
            | VariableKind::LocalConst { .. }
            | VariableKind::Versioned { .. }
            | VariableKind::Matrix { .. }
            | VariableKind::Barrier { .. }
            | VariableKind::Pipeline { .. } => {
                self.variable_uniformity.get(&var).copied().unwrap_or(true)
            }
            VariableKind::TensorMap(_) => true,
        }
    }

    pub fn is_block_uniform(&self, block: NodeIndex) -> bool {
        self.block_uniformity.get(&block).copied().unwrap_or(true)
    }
}
