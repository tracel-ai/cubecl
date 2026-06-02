use core::fmt::Debug;

use alloc::vec::Vec;

use cubecl_ir::{
    Branch, CoopMma, GlobalState, Instruction, Marker, NonSemantic, Operation, OperationReflect,
    Operator, Scope, TensorIndexingOps, TmaOps, Variable,
};
use derive_more::{Deref, DerefMut};

use crate::post_processing::{analysis_helper::GlobalAnalyses, util::AtomicCounter};

/// Visitor that operates on an instruction level. Useful for passes that only need to recursively
/// traverse the scopes and don't care about control flow.
///
/// The `changes` counter should be incremented on any change, unless the pass is a unique one-time
/// pass. It's used to determine when to end a fixed-point optimization loop.
pub trait InstructionVisitor: Debug {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        changes: &AtomicCounter,
    ) -> Vec<Instruction>;

    fn visit_scope(&mut self, scope: &Scope, analyses: &GlobalAnalyses, changes: &AtomicCounter) {
        visit_scope(self, scope, analyses, changes);
    }
}

pub fn visit_scope<T: InstructionVisitor + ?Sized>(
    visitor: &mut T,
    scope: &Scope,
    analyses: &GlobalAnalyses,
    changes: &AtomicCounter,
) {
    let instructions = scope.take_instructions();
    let mut new_instructions = Vec::with_capacity(instructions.len());
    for inst in instructions {
        if let Operation::Branch(branch) = &inst.operation {
            match branch {
                Branch::If(op) => {
                    visitor.visit_scope(&op.scope, analyses, changes);
                }
                Branch::IfElse(op) => {
                    visitor.visit_scope(&op.scope_if, analyses, changes);
                    visitor.visit_scope(&op.scope_else, analyses, changes);
                }
                Branch::Switch(op) => {
                    for (_, case) in &op.cases {
                        visitor.visit_scope(case, analyses, changes);
                    }
                    visitor.visit_scope(&op.scope_default, analyses, changes);
                }
                Branch::RangeLoop(op) => {
                    visitor.visit_scope(&op.scope, analyses, changes);
                }
                Branch::Loop(op) => {
                    visitor.visit_scope(&op.scope, analyses, changes);
                }
                _ => {}
            }
        }

        new_instructions.extend(visitor.visit_instruction(
            inst,
            &scope.global_state,
            analyses,
            changes,
        ));
    }

    scope.register_all(new_instructions);
}

#[derive(Deref, DerefMut)]
pub struct Visitor<T>(pub T);

impl<T> Visitor<T> {
    pub fn visit_instruction(
        &mut self,
        inst: &mut Instruction,
        analyses: &GlobalAnalyses,
        visit_read: impl FnMut(&mut T, &mut Variable),
        mut visit_write: impl FnMut(&mut T, &mut Variable),
    ) {
        self.visit_operation(&mut inst.operation, analyses, visit_read);

        for ptr in inst.operation.write_pointers() {
            if let Some(source) = analyses.ptr_source().get_mut(&ptr.kind) {
                visit_write(self, source);
            }
        }

        self.visit_out(&mut inst.out, visit_write);
    }

    pub fn visit_out(
        &mut self,
        out: &mut Option<Variable>,
        mut visit_write: impl FnMut(&mut T, &mut Variable),
    ) {
        if let Some(out) = out {
            visit_write(self, out)
        }
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operation(
        &mut self,
        op: &mut Operation,
        analyses: &GlobalAnalyses,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        for ptr in op.read_pointers() {
            if let Some(source) = analyses.ptr_source().get_mut(&ptr.kind) {
                visit_read(self, source);
            }
        }

        match op {
            Operation::Marker(Marker::Free(_)) => {}
            Operation::CoopMma(coop_mma) => self.visit_cmma(coop_mma, visit_read),
            Operation::Branch(branch) => self.visit_branch(branch, visit_read),
            Operation::Tma(tma_ops) => self.visit_tma(tma_ops, visit_read),
            Operation::TensorIndexing(tensor_ops) => self.visit_tensor_ops(tensor_ops, visit_read),
            Operation::NonSemantic(non_semantic) => {
                self.visit_nonsemantic(non_semantic, visit_read)
            }
            Operation::Operator(Operator::ReadBuiltin(_)) => {}
            op => {
                if let Some(args) = op.args_mut() {
                    for arg in args {
                        visit_read(self, arg);
                    }
                } else {
                    panic!("Found op {op:?} which doesn't reflect. Needs special handling.");
                }
            }
        }
    }

    /// Visit a control flow finisher with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_branch(
        &mut self,
        op: &mut Branch,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match op {
            Branch::If(if_) => visit_read(self, &mut if_.cond),
            Branch::IfElse(if_else) => visit_read(self, &mut if_else.cond),
            Branch::Switch(switch) => visit_read(self, &mut switch.value),
            Branch::RangeLoop(range_loop) => {
                visit_read(self, &mut range_loop.start);
                visit_read(self, &mut range_loop.end);
            }
            Branch::Loop(_) | Branch::Return | Branch::Break | Branch::Unreachable => {}
        }
    }

    fn visit_cmma(
        &mut self,
        cmma: &mut CoopMma,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match cmma {
            CoopMma::Fill { value } => {
                visit_read(self, value);
            }
            CoopMma::Load {
                ptr,
                stride,
                layout: _,
            } => {
                visit_read(self, ptr);
                visit_read(self, stride);
            }
            CoopMma::LoadTensor {
                buffer,
                layout,
                view,
            } => {
                visit_read(self, buffer);
                visit_read(self, layout);
                if let Some(view) = view {
                    visit_read(self, view);
                }
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => {
                visit_read(self, mat_a);
                visit_read(self, mat_b);
                visit_read(self, mat_c);
            }
            CoopMma::Store {
                mat,
                stride,
                destination,
                layout: _,
            } => {
                visit_read(self, mat);
                visit_read(self, stride);
                visit_read(self, destination);
            }
            CoopMma::StoreTensor { mat, layout, view } => {
                visit_read(self, mat);
                visit_read(self, layout);
                if let Some(view) = view {
                    visit_read(self, view);
                }
            }
            CoopMma::Cast { input } => {
                visit_read(self, input);
            }
            CoopMma::RowIndex { lane_id, i, .. } => {
                visit_read(self, lane_id);
                visit_read(self, i);
            }
            CoopMma::ColIndex { lane_id, i, .. } => {
                visit_read(self, lane_id);
                visit_read(self, i);
            }
            CoopMma::LoadMatrix { ptr, .. } => {
                visit_read(self, ptr);
            }
            CoopMma::StoreMatrix {
                registers,
                destination,
                ..
            } => {
                visit_read(self, registers);
                visit_read(self, destination);
            }
            CoopMma::ExecuteManual {
                registers_a,
                registers_b,
                registers_c,
                ..
            } => {
                visit_read(self, registers_a);
                visit_read(self, registers_b);
                visit_read(self, registers_c);
            }
            CoopMma::ExecuteScaled {
                registers_a,
                registers_b,
                registers_c,
                scales_a,
                scales_b,
                ..
            } => {
                visit_read(self, registers_a);
                visit_read(self, registers_b);
                visit_read(self, registers_c);
                visit_read(self, scales_a);
                visit_read(self, scales_b);
            }
            CoopMma::ExecuteElementwise { matrix, .. } => {
                visit_read(self, matrix);
            }
        }
    }

    fn visit_tma(
        &mut self,
        tma_ops: &mut TmaOps,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match tma_ops {
            TmaOps::TmaStore {
                source,
                coordinates,
            } => {
                visit_read(self, source);
                for coord in coordinates {
                    visit_read(self, coord)
                }
            }
            TmaOps::CommitGroup | TmaOps::WaitGroup { .. } | TmaOps::WaitGroupRead { .. } => {}
        }
    }

    fn visit_tensor_ops(
        &mut self,
        tensor_ops: &mut TensorIndexingOps,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match tensor_ops {
            TensorIndexingOps::CreateLayout {
                shape,
                strides,
                clamp_mode: _,
            } => {
                for s in shape {
                    visit_read(self, s);
                }
                for s in strides.iter_mut().flatten() {
                    visit_read(self, s);
                }
            }
            TensorIndexingOps::CreateView => {}
            TensorIndexingOps::Slice {
                layout,
                offsets,
                shape,
            } => {
                visit_read(self, layout);
                for o in offsets {
                    visit_read(self, o);
                }
                for s in shape {
                    visit_read(self, s);
                }
            }
        }
    }

    fn visit_nonsemantic(
        &mut self,
        non_semantic: &mut NonSemantic,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match non_semantic {
            NonSemantic::Comment { .. }
            | NonSemantic::EnterDebugScope
            | NonSemantic::ExitDebugScope => {}
            NonSemantic::Print { args, .. } => {
                for arg in args {
                    visit_read(self, arg);
                }
            }
        }
    }
}
