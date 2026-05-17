use core::fmt::Debug;

use alloc::vec::Vec;

use cubecl_ir::{
    Arithmetic, AtomicBinaryOperands, AtomicOp, BarrierOps, BinaryOperands, Bitwise, Branch,
    Comparison, CoopMma, GlobalState, Instruction, Marker, Memory, Metadata, NonSemantic,
    Operation, OperationReflect, Operator, Plane, Scope, TensorIndexingOps, TmaOps, UnaryOperands,
    Variable,
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
            Operation::Copy(variable) => visit_read(self, variable),
            Operation::WorkgroupUniformLoad(variable) => visit_read(self, variable),
            Operation::Memory(memory) => self.visit_memory(memory, visit_read),
            Operation::Arithmetic(arithmetic) => self.visit_arithmetic(arithmetic, visit_read),
            Operation::Comparison(comparison) => self.visit_compare(comparison, visit_read),
            Operation::Bitwise(bitwise) => self.visit_bitwise(bitwise, visit_read),
            Operation::Operator(operator) => self.visit_operator(operator, visit_read),
            Operation::Atomic(atomic) => self.visit_atomic(atomic, visit_read),
            Operation::Metadata(meta) => self.visit_meta(meta, visit_read),
            // Sync has no outputs
            Operation::Synchronization(_) => {}
            Operation::Plane(plane) => self.visit_plane(plane, visit_read),
            Operation::CoopMma(coop_mma) => self.visit_cmma(coop_mma, visit_read),
            Operation::Branch(branch) => self.visit_branch(branch, visit_read),
            Operation::Barrier(barrier_ops) => self.visit_barrier(barrier_ops, visit_read),
            Operation::Tma(tma_ops) => self.visit_tma(tma_ops, visit_read),
            Operation::TensorIndexing(tensor_ops) => self.visit_tensor_ops(tensor_ops, visit_read),
            Operation::NonSemantic(non_semantic) => {
                self.visit_nonsemantic(non_semantic, visit_read)
            }
            Operation::Marker(Marker::Free(_)) => {}
            Operation::Marker(Marker::DummyRead(variable)) => visit_read(self, variable),
            Operation::ConstructAggregate(variables) => {
                for variable in variables {
                    visit_read(self, variable);
                }
            }
            Operation::ExtractAggregateField(op) => visit_read(self, &mut op.aggregate),
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

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_arithmetic(
        &mut self,
        op: &mut Arithmetic,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match op {
            Arithmetic::Fma(fma_operands) => {
                visit_read(self, &mut fma_operands.a);
                visit_read(self, &mut fma_operands.b);
                visit_read(self, &mut fma_operands.c);
            }
            Arithmetic::Add(binary_operands)
            | Arithmetic::SaturatingAdd(binary_operands)
            | Arithmetic::Sub(binary_operands)
            | Arithmetic::SaturatingSub(binary_operands)
            | Arithmetic::Mul(binary_operands)
            | Arithmetic::Div(binary_operands)
            | Arithmetic::Powf(binary_operands)
            | Arithmetic::Powi(binary_operands)
            | Arithmetic::Hypot(binary_operands)
            | Arithmetic::Rhypot(binary_operands)
            | Arithmetic::ModFloor(binary_operands)
            | Arithmetic::Max(binary_operands)
            | Arithmetic::Min(binary_operands)
            | Arithmetic::Rem(binary_operands)
            | Arithmetic::Dot(binary_operands)
            | Arithmetic::MulHi(binary_operands)
            | Arithmetic::ArcTan2(binary_operands) => self.visit_binop(binary_operands, visit_read),

            Arithmetic::Abs(unary_operands)
            | Arithmetic::Exp(unary_operands)
            | Arithmetic::Log(unary_operands)
            | Arithmetic::Log1p(unary_operands)
            | Arithmetic::Cos(unary_operands)
            | Arithmetic::Sin(unary_operands)
            | Arithmetic::Tan(unary_operands)
            | Arithmetic::Tanh(unary_operands)
            | Arithmetic::Sinh(unary_operands)
            | Arithmetic::Cosh(unary_operands)
            | Arithmetic::ArcCos(unary_operands)
            | Arithmetic::ArcSin(unary_operands)
            | Arithmetic::ArcTan(unary_operands)
            | Arithmetic::ArcSinh(unary_operands)
            | Arithmetic::ArcCosh(unary_operands)
            | Arithmetic::ArcTanh(unary_operands)
            | Arithmetic::Degrees(unary_operands)
            | Arithmetic::Radians(unary_operands)
            | Arithmetic::Sqrt(unary_operands)
            | Arithmetic::InverseSqrt(unary_operands)
            | Arithmetic::Round(unary_operands)
            | Arithmetic::Floor(unary_operands)
            | Arithmetic::Ceil(unary_operands)
            | Arithmetic::Trunc(unary_operands)
            | Arithmetic::Erf(unary_operands)
            | Arithmetic::Recip(unary_operands)
            | Arithmetic::Neg(unary_operands)
            | Arithmetic::Magnitude(unary_operands)
            | Arithmetic::Normalize(unary_operands)
            | Arithmetic::VectorSum(unary_operands) => self.visit_unop(unary_operands, visit_read),

            Arithmetic::Clamp(clamp_operands) => {
                visit_read(self, &mut clamp_operands.input);
                visit_read(self, &mut clamp_operands.min_value);
                visit_read(self, &mut clamp_operands.max_value);
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_compare(
        &mut self,
        op: &mut Comparison,
        visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match op {
            Comparison::Equal(binary_operands)
            | Comparison::NotEqual(binary_operands)
            | Comparison::LowerEqual(binary_operands)
            | Comparison::Greater(binary_operands)
            | Comparison::Lower(binary_operands)
            | Comparison::GreaterEqual(binary_operands) => {
                self.visit_binop(binary_operands, visit_read)
            }
            Comparison::IsNan(unary_operands) | Comparison::IsInf(unary_operands) => {
                self.visit_unop(unary_operands, visit_read)
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_bitwise(
        &mut self,
        op: &mut Bitwise,
        visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match op {
            Bitwise::BitwiseAnd(binary_operands)
            | Bitwise::BitwiseOr(binary_operands)
            | Bitwise::BitwiseXor(binary_operands)
            | Bitwise::ShiftLeft(binary_operands)
            | Bitwise::ShiftRight(binary_operands) => self.visit_binop(binary_operands, visit_read),

            Bitwise::CountOnes(unary_operands)
            | Bitwise::BitwiseNot(unary_operands)
            | Bitwise::ReverseBits(unary_operands)
            | Bitwise::LeadingZeros(unary_operands)
            | Bitwise::TrailingZeros(unary_operands)
            | Bitwise::FindFirstSet(unary_operands) => self.visit_unop(unary_operands, visit_read),
        }
    }

    pub fn visit_memory(
        &mut self,
        memory: &mut Memory,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match memory {
            Memory::Reference(variable) => visit_read(self, variable),
            Memory::Index(index_operands) => {
                visit_read(self, &mut index_operands.list);
                visit_read(self, &mut index_operands.index);
            }
            Memory::Load(variable) => {
                visit_read(self, variable);
            }
            Memory::Store(op) => {
                visit_read(self, &mut op.ptr);
                visit_read(self, &mut op.value);
            }
            Memory::CopyMemory(op) => {
                visit_read(self, &mut op.source);
                visit_read(self, &mut op.target);
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operator(
        &mut self,
        op: &mut Operator,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match op {
            Operator::And(binary_operands)
            | Operator::Or(binary_operands)
            | Operator::ExtractComponent(binary_operands) => {
                self.visit_binop(binary_operands, visit_read)
            }
            Operator::Not(unary_operands)
            | Operator::Cast(unary_operands)
            | Operator::Reinterpret(unary_operands) => self.visit_unop(unary_operands, visit_read),
            Operator::InitVector(vector_init_operator) => {
                for input in &mut vector_init_operator.inputs {
                    visit_read(self, input)
                }
            }
            Operator::InsertComponent(vector_insert_operands) => {
                visit_read(self, &mut vector_insert_operands.vector);
                visit_read(self, &mut vector_insert_operands.index);
                visit_read(self, &mut vector_insert_operands.value);
            }
            Operator::Select(select_operands) => {
                visit_read(self, &mut select_operands.cond);
                visit_read(self, &mut select_operands.then);
                visit_read(self, &mut select_operands.or_else);
            }
        }
    }

    fn visit_atomic(
        &mut self,
        atomic: &mut AtomicOp,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match atomic {
            AtomicOp::Add(atomic_binary_operands)
            | AtomicOp::Sub(atomic_binary_operands)
            | AtomicOp::Max(atomic_binary_operands)
            | AtomicOp::Min(atomic_binary_operands)
            | AtomicOp::And(atomic_binary_operands)
            | AtomicOp::Or(atomic_binary_operands)
            | AtomicOp::Xor(atomic_binary_operands)
            | AtomicOp::Swap(atomic_binary_operands) => {
                self.visit_atomic_binop(atomic_binary_operands, visit_read);
            }
            AtomicOp::Load(ptr) => {
                visit_read(self, ptr);
            }
            AtomicOp::Store(store) => {
                visit_read(self, &mut store.ptr);
                visit_read(self, &mut store.value);
            }
            AtomicOp::CompareAndSwap(op) => {
                visit_read(self, &mut op.cmp);
                visit_read(self, &mut op.cmp);
                visit_read(self, &mut op.val);
            }
        }
    }
    fn visit_meta(
        &mut self,
        metadata: &mut Metadata,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match metadata {
            Metadata::BufferLength { var } => {
                visit_read(self, var);
            }
            Metadata::Stride { dim, var } => {
                visit_read(self, dim);
                visit_read(self, var);
            }
            Metadata::Shape { dim, var } => {
                visit_read(self, dim);
                visit_read(self, var);
            }
        }
    }

    fn visit_plane(&mut self, plane: &mut Plane, visit_read: impl FnMut(&mut T, &mut Variable)) {
        match plane {
            Plane::Elect => {}
            Plane::Broadcast(binary_operands)
            | Plane::Shuffle(binary_operands)
            | Plane::ShuffleXor(binary_operands)
            | Plane::ShuffleUp(binary_operands)
            | Plane::ShuffleDown(binary_operands) => self.visit_binop(binary_operands, visit_read),
            Plane::All(unary_operands)
            | Plane::Any(unary_operands)
            | Plane::Sum(unary_operands)
            | Plane::InclusiveSum(unary_operands)
            | Plane::ExclusiveSum(unary_operands)
            | Plane::Prod(unary_operands)
            | Plane::InclusiveProd(unary_operands)
            | Plane::ExclusiveProd(unary_operands)
            | Plane::Min(unary_operands)
            | Plane::Max(unary_operands)
            | Plane::Ballot(unary_operands) => self.visit_unop(unary_operands, visit_read),
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

    fn visit_barrier(
        &mut self,
        barrier_ops: &mut BarrierOps,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        match barrier_ops {
            BarrierOps::Declare { barrier } => visit_read(self, barrier),
            BarrierOps::Init {
                barrier,
                is_elected,
                arrival_count,
                ..
            } => {
                visit_read(self, barrier);
                visit_read(self, is_elected);
                visit_read(self, arrival_count);
            }
            BarrierOps::InitManual {
                barrier,
                arrival_count,
            } => {
                visit_read(self, barrier);
                visit_read(self, arrival_count);
            }
            BarrierOps::MemCopyAsync {
                barrier,
                source,
                destination,
                source_length,
            } => {
                visit_read(self, barrier);
                visit_read(self, source_length);
                visit_read(self, source);
                visit_read(self, destination);
            }
            BarrierOps::MemCopyAsyncCooperative {
                barrier,
                source,
                destination,
                source_length,
            } => {
                visit_read(self, barrier);
                visit_read(self, source_length);
                visit_read(self, source);
                visit_read(self, destination);
            }
            BarrierOps::CopyAsync {
                source,
                source_length,
                destination,
                ..
            } => {
                visit_read(self, source_length);
                visit_read(self, source);
                visit_read(self, destination);
            }
            BarrierOps::MemCopyAsyncTx {
                barrier,
                source,
                destination,
                source_length,
            } => {
                visit_read(self, barrier);
                visit_read(self, source_length);
                visit_read(self, source);
                visit_read(self, destination);
            }
            BarrierOps::TmaLoad {
                barrier,
                tensor_map,
                destination,
                indices,
            } => {
                visit_read(self, barrier);
                visit_read(self, tensor_map);
                visit_read(self, destination);
                for index in indices {
                    visit_read(self, index);
                }
            }
            BarrierOps::TmaLoadIm2col {
                barrier,
                tensor_map,
                destination,
                indices,
                offsets,
            } => {
                visit_read(self, barrier);
                visit_read(self, tensor_map);
                visit_read(self, destination);
                for index in indices {
                    visit_read(self, index);
                }
                for offset in offsets {
                    visit_read(self, offset);
                }
            }
            BarrierOps::ArriveAndWait { barrier } => visit_read(self, barrier),
            BarrierOps::Arrive { barrier } => visit_read(self, barrier),
            BarrierOps::ArriveTx {
                barrier,
                arrive_count_update,
                transaction_count_update,
            } => {
                visit_read(self, barrier);
                visit_read(self, arrive_count_update);
                visit_read(self, transaction_count_update);
            }
            BarrierOps::CommitCopyAsync { barrier } => visit_read(self, barrier),
            BarrierOps::ExpectTx {
                barrier,
                transaction_count_update,
            } => {
                visit_read(self, barrier);
                visit_read(self, transaction_count_update);
            }
            BarrierOps::Wait { barrier, token } => {
                visit_read(self, barrier);
                visit_read(self, token);
            }
            BarrierOps::WaitParity { barrier, phase } => {
                visit_read(self, barrier);
                visit_read(self, phase);
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

    fn visit_unop(
        &mut self,
        unop: &mut UnaryOperands,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        visit_read(self, &mut unop.input);
    }

    fn visit_binop(
        &mut self,
        binop: &mut BinaryOperands,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        visit_read(self, &mut binop.lhs);
        visit_read(self, &mut binop.rhs);
    }

    fn visit_atomic_binop(
        &mut self,
        binop: &mut AtomicBinaryOperands,
        mut visit_read: impl FnMut(&mut T, &mut Variable),
    ) {
        visit_read(self, &mut binop.ptr);
        visit_read(self, &mut binop.value);
    }
}
