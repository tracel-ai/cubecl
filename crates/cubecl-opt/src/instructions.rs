use cubecl_ir::{
    CoopMma, Instruction, Marker, Metadata, NonSemantic, Operation, OperationReflect, Operator,
    TensorIndexingOps, TmaOps, Variable,
};

use crate::{ControlFlow, Function, GlobalState, analyses::pointer_source::PointerSource};

impl Function {
    pub fn visit_out(
        &mut self,
        var: &mut Option<Variable>,
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        if let Some(out) = var {
            visit_write(self, out);
        }
    }

    /// Visit an instruction with only a write visitor. Visits both `out` and any pointer writes.
    pub fn visit_instruction_write(
        &mut self,
        state: &GlobalState,
        inst: &mut Instruction,
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        let pointer_source = self.analysis::<PointerSource>(state);
        for ptr in inst.operation.write_pointers() {
            if let Some(source) = pointer_source.borrow_mut().get_mut(&ptr) {
                visit_write(self, source);
            }
        }

        // This is hacky, because semantics for insert are very awkward.
        if let Operation::Operator(Operator::InsertComponent(_)) = &mut inst.operation
            && let Some(source) = pointer_source.borrow_mut().get_mut(&inst.out())
        {
            visit_write(self, source);
        }

        self.visit_out(&mut inst.out, visit_write);
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_instruction(
        &mut self,
        state: &GlobalState,
        inst: &mut Instruction,
        visit_read: impl FnMut(&mut Self, &mut Variable),
        visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        self.visit_operation(state, &mut inst.operation, visit_read);
        self.visit_instruction_write(state, inst, visit_write);
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operation(
        &mut self,
        state: &GlobalState,
        op: &mut Operation,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        let pointer_source = self.analysis::<PointerSource>(state);
        for ptr in op.read_pointers() {
            if let Some(source) = pointer_source.borrow_mut().get_mut(&ptr) {
                visit_read(self, source);
            }
        }

        match op {
            Operation::Marker(Marker::Free(_)) => {}
            Operation::Metadata(meta) => self.visit_meta(meta, visit_read),
            Operation::CoopMma(coop_mma) => self.visit_cmma(state, coop_mma, visit_read),
            Operation::Branch(_) => unreachable!(),
            Operation::Tma(tma_ops) => self.visit_tma(tma_ops, visit_read),
            Operation::TensorIndexing(tensor_ops) => self.visit_tensor_ops(tensor_ops, visit_read),
            Operation::NonSemantic(non_semantic) => {
                self.visit_nonsemantic(non_semantic, visit_read)
            }
            op => {
                if let Some(args) = op.args_mut() {
                    for arg in args {
                        visit_read(self, arg);
                    }
                } else {
                    panic!("Found op {op} which doesn't reflect. Needs special handling.");
                }
            }
        }
    }

    /// Visit a control flow finisher with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_control_flow(
        &mut self,
        op: &mut ControlFlow,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            ControlFlow::IfElse { cond, .. } => visit_read(self, cond),
            ControlFlow::Switch { value, .. } => visit_read(self, value),
            ControlFlow::Loop { .. } => {}
            ControlFlow::LoopBreak { break_cond, .. } => visit_read(self, break_cond),
            ControlFlow::Return { value } => {
                if let Some(value) = value {
                    visit_read(self, value);
                }
            }
            ControlFlow::Unreachable | ControlFlow::None => {}
        }
    }

    fn visit_meta(
        &mut self,
        metadata: &mut Metadata,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        // Don't count buffer as a read, since it's actually the info buffer that's read.
        match metadata {
            Metadata::BufferLength { .. } => {}
            Metadata::Stride { dim, .. } => {
                visit_read(self, dim);
            }
            Metadata::Shape { dim, .. } => {
                visit_read(self, dim);
            }
        }
    }

    fn visit_cmma(
        &mut self,
        state: &GlobalState,
        cmma: &mut CoopMma,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
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
            CoopMma::StoreMatrix { registers, .. } => {
                visit_read(self, registers);
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
            CoopMma::ExecuteElementwise { matrix, op } => {
                visit_read(self, matrix);
                let func = &state.extra_functions[op];
                for mut capture in func.implicit_params.clone() {
                    visit_read(self, &mut capture)
                }
            }
        }
    }

    fn visit_tma(
        &mut self,
        tma_ops: &mut TmaOps,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
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
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
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
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
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
