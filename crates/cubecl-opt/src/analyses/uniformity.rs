use cubecl_ir::{Builtin, Operation, OperationReflect, Plane, Variable, VariableKind};
use petgraph::{graph::EdgeIndex, visit::EdgeRef};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

use crate::{ControlFlow, NodeIndex, Optimizer};

use super::Analysis;

#[derive(Default)]
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
        self.analyze_block(opt, root);
    }

    fn analyze_block(&mut self, opt: &Optimizer, block_id: NodeIndex) {
        let block = opt.block(block_id);
        let block_uniform = self.block_uniformity[&block_id];

        for phi in block.phi_nodes.borrow().iter() {
            let uniform = phi.entries.iter().all(|entry| {
                let block_uniform = self.is_block_uniform(entry.block);
                let value_uniform = self.is_var_uniform(entry.value);
                block_uniform && value_uniform
            });
            self.mark_uniformity(phi.out, uniform && block_uniform);
        }

        for inst in block.ops.borrow().values() {
            if inst.out.is_none() {
                continue;
            }
            let out = inst.out.unwrap();
            match &inst.operation {
                Operation::Plane(plane) => match plane {
                    // Elect returns true on only one unit, so it's always non-uniform
                    Plane::Elect => self.mark_uniformity(out, false),
                    // Reductions are always uniform
                    Plane::Sum(_)
                    | Plane::Prod(_)
                    | Plane::Min(_)
                    | Plane::Max(_)
                    | Plane::All(_)
                    | Plane::Any(_) => self.mark_uniformity(out, true),
                    // Broadcast maps to shuffle or broadcast, if id or value is uniform, so will
                    // the output, otherwise not.
                    Plane::Broadcast(op) => {
                        let input_uniform =
                            self.is_var_uniform(op.lhs) || self.is_var_uniform(op.rhs);
                        self.mark_uniformity(out, input_uniform && block_uniform);
                    }
                },
                op => {
                    let is_uniform =
                        op.is_pure() && self.is_all_uniform(op.args()) && block_uniform;
                    self.mark_uniformity(out, is_uniform);
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
                self.analyze_block(opt, *then);
                self.analyze_block(opt, *or_else);
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
                self.analyze_block(opt, edge.target());
            }
        }
    }

    fn mark_uniformity(&mut self, var: Variable, value: bool) {
        self.variable_uniformity.insert(var, value);
    }

    fn is_all_uniform(&self, args: Option<SmallVec<[Variable; 4]>>) -> bool {
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
                | Builtin::CubeDim
                | Builtin::CubeDimX
                | Builtin::CubeDimY
                | Builtin::CubeDimZ
                | Builtin::CubeCount
                | Builtin::CubeCountX
                | Builtin::CubeCountY
                | Builtin::CubeCountZ
                | Builtin::PlaneDim => true,
            },

            VariableKind::LocalArray { .. }
            | VariableKind::LocalMut { .. }
            | VariableKind::LocalConst { .. }
            | VariableKind::Versioned { .. }
            | VariableKind::Matrix { .. }
            | VariableKind::Slice { .. }
            | VariableKind::Pipeline { .. } => {
                self.variable_uniformity.get(&var).copied().unwrap_or(false)
            }
        }
    }

    pub fn is_block_uniform(&self, block: NodeIndex) -> bool {
        self.block_uniformity.get(&block).copied().unwrap_or(false)
    }
}
