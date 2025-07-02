use cubecl_ir::{
    Arithmetic, AtomicOp, BarrierOps, BinaryOperator, Bitwise, Comparison, CoopMma, Instruction,
    Metadata, NonSemantic, Operation, Operator, Plane, TmaOps, UnaryOperator, Variable,
};

use super::Optimizer;

impl Optimizer {
    pub fn visit_out(
        &mut self,
        var: &mut Option<Variable>,
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        if let Some(out) = var {
            visit_write(self, out);
        }
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_instruction(
        &mut self,
        inst: &mut Instruction,
        visit_read: impl FnMut(&mut Self, &mut Variable),
        visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        self.visit_out(&mut inst.out, visit_write);
        self.visit_operation(&mut inst.operation, &mut inst.out, visit_read);
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operation(
        &mut self,
        op: &mut Operation,
        out: &mut Option<Variable>,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Operation::Copy(variable) => visit_read(self, variable),
            Operation::Arithmetic(arithmetic) => self.visit_arithmetic(arithmetic, visit_read),
            Operation::Comparison(comparison) => self.visit_compare(comparison, visit_read),
            Operation::Bitwise(bitwise) => self.visit_bitwise(bitwise, visit_read),
            Operation::Operator(operator) => self.visit_operator(operator, visit_read),
            Operation::Atomic(atomic) => self.visit_atomic(atomic, out, visit_read),
            Operation::Metadata(meta) => self.visit_meta(meta, visit_read),
            // Sync has no outputs
            Operation::Synchronization(_) => {}
            Operation::Plane(plane) => self.visit_plane(plane, visit_read),
            Operation::CoopMma(coop_mma) => self.visit_cmma(coop_mma, visit_read),
            Operation::Branch(_) => unreachable!(),
            Operation::Barrier(barrier_ops) => self.visit_barrier(barrier_ops, visit_read),
            Operation::Tma(tma_ops) => self.visit_tma(tma_ops, visit_read),
            Operation::NonSemantic(non_semantic) => {
                self.visit_nonsemantic(non_semantic, visit_read)
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_arithmetic(
        &mut self,
        op: &mut Arithmetic,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Arithmetic::Fma(fma_operator) => {
                visit_read(self, &mut fma_operator.a);
                visit_read(self, &mut fma_operator.b);
                visit_read(self, &mut fma_operator.c);
            }
            Arithmetic::Add(binary_operator)
            | Arithmetic::Sub(binary_operator)
            | Arithmetic::Mul(binary_operator)
            | Arithmetic::Div(binary_operator)
            | Arithmetic::Powf(binary_operator)
            | Arithmetic::Modulo(binary_operator)
            | Arithmetic::Max(binary_operator)
            | Arithmetic::Min(binary_operator)
            | Arithmetic::Remainder(binary_operator)
            | Arithmetic::Dot(binary_operator)
            | Arithmetic::MulHi(binary_operator) => self.visit_binop(binary_operator, visit_read),

            Arithmetic::Abs(unary_operator)
            | Arithmetic::Exp(unary_operator)
            | Arithmetic::Log(unary_operator)
            | Arithmetic::Log1p(unary_operator)
            | Arithmetic::Cos(unary_operator)
            | Arithmetic::Sin(unary_operator)
            | Arithmetic::Tanh(unary_operator)
            | Arithmetic::Sqrt(unary_operator)
            | Arithmetic::Round(unary_operator)
            | Arithmetic::Floor(unary_operator)
            | Arithmetic::Ceil(unary_operator)
            | Arithmetic::Erf(unary_operator)
            | Arithmetic::Recip(unary_operator)
            | Arithmetic::Neg(unary_operator)
            | Arithmetic::Magnitude(unary_operator)
            | Arithmetic::Normalize(unary_operator) => self.visit_unop(unary_operator, visit_read),

            Arithmetic::Clamp(clamp_operator) => {
                visit_read(self, &mut clamp_operator.input);
                visit_read(self, &mut clamp_operator.min_value);
                visit_read(self, &mut clamp_operator.max_value);
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_compare(
        &mut self,
        op: &mut Comparison,
        visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Comparison::Equal(binary_operator)
            | Comparison::NotEqual(binary_operator)
            | Comparison::LowerEqual(binary_operator)
            | Comparison::Greater(binary_operator)
            | Comparison::Lower(binary_operator)
            | Comparison::GreaterEqual(binary_operator) => {
                self.visit_binop(binary_operator, visit_read)
            }
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_bitwise(
        &mut self,
        op: &mut Bitwise,
        visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Bitwise::BitwiseAnd(binary_operator)
            | Bitwise::BitwiseOr(binary_operator)
            | Bitwise::BitwiseXor(binary_operator)
            | Bitwise::ShiftLeft(binary_operator)
            | Bitwise::ShiftRight(binary_operator) => self.visit_binop(binary_operator, visit_read),

            Bitwise::CountOnes(unary_operator)
            | Bitwise::BitwiseNot(unary_operator)
            | Bitwise::ReverseBits(unary_operator)
            | Bitwise::LeadingZeros(unary_operator)
            | Bitwise::FindFirstSet(unary_operator) => self.visit_unop(unary_operator, visit_read),
        }
    }

    /// Visit an operator with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operator(
        &mut self,
        op: &mut Operator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Operator::And(binary_operator) | Operator::Or(binary_operator) => {
                self.visit_binop(binary_operator, visit_read)
            }
            Operator::Not(unary_operator)
            | Operator::Cast(unary_operator)
            | Operator::Reinterpret(unary_operator) => self.visit_unop(unary_operator, visit_read),
            Operator::Index(index_operator) | Operator::UncheckedIndex(index_operator) => {
                visit_read(self, &mut index_operator.list);
                visit_read(self, &mut index_operator.index);
            }
            Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op) => {
                visit_read(self, &mut op.index);
                visit_read(self, &mut op.value);
            }
            Operator::InitLine(line_init_operator) => {
                for input in &mut line_init_operator.inputs {
                    visit_read(self, input)
                }
            }
            Operator::CopyMemory(copy_operator) => {
                visit_read(self, &mut copy_operator.input);
                visit_read(self, &mut copy_operator.in_index);
                visit_read(self, &mut copy_operator.out_index);
            }
            Operator::CopyMemoryBulk(copy_bulk_operator) => {
                visit_read(self, &mut copy_bulk_operator.input);
                visit_read(self, &mut copy_bulk_operator.in_index);
                visit_read(self, &mut copy_bulk_operator.out_index);
            }
            Operator::Select(select) => {
                visit_read(self, &mut select.cond);
                visit_read(self, &mut select.then);
                visit_read(self, &mut select.or_else);
            }
        }
    }

    fn visit_atomic(
        &mut self,
        atomic: &mut AtomicOp,
        out: &mut Option<Variable>,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match atomic {
            AtomicOp::Add(binary_operator)
            | AtomicOp::Sub(binary_operator)
            | AtomicOp::Max(binary_operator)
            | AtomicOp::Min(binary_operator)
            | AtomicOp::And(binary_operator)
            | AtomicOp::Or(binary_operator)
            | AtomicOp::Xor(binary_operator)
            | AtomicOp::Swap(binary_operator) => {
                self.visit_binop(binary_operator, visit_read);
            }
            AtomicOp::Load(unary_operator) => {
                self.visit_unop(unary_operator, visit_read);
            }
            AtomicOp::Store(unary_operator) => {
                visit_read(self, out.as_mut().unwrap());
                self.visit_unop(unary_operator, visit_read);
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
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match metadata {
            Metadata::Rank { var } => {
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
            Metadata::Length { var } => {
                visit_read(self, var);
            }
            Metadata::BufferLength { var } => {
                visit_read(self, var);
            }
        }
    }

    fn visit_plane(&mut self, plane: &mut Plane, visit_read: impl FnMut(&mut Self, &mut Variable)) {
        match plane {
            Plane::Elect => {}
            Plane::Broadcast(binary_operator) => self.visit_binop(binary_operator, visit_read),
            Plane::All(unary_operator)
            | Plane::Any(unary_operator)
            | Plane::Sum(unary_operator)
            | Plane::InclusiveSum(unary_operator)
            | Plane::ExclusiveSum(unary_operator)
            | Plane::Prod(unary_operator)
            | Plane::InclusiveProd(unary_operator)
            | Plane::ExclusiveProd(unary_operator)
            | Plane::Min(unary_operator)
            | Plane::Max(unary_operator)
            | Plane::Ballot(unary_operator) => self.visit_unop(unary_operator, visit_read),
        }
    }

    fn visit_cmma(
        &mut self,
        cmma: &mut CoopMma,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match cmma {
            CoopMma::Fill { value } => {
                visit_read(self, value);
            }
            CoopMma::Load {
                value,
                stride,
                offset,
                layout: _,
            } => {
                visit_read(self, value);
                visit_read(self, stride);
                visit_read(self, offset);
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
                offset,
                layout: _,
            } => {
                visit_read(self, mat);
                visit_read(self, stride);
                visit_read(self, offset);
            }
            CoopMma::Cast { input } => {
                visit_read(self, input);
            }
        }
    }

    fn visit_barrier(
        &mut self,
        barrier_ops: &mut BarrierOps,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match barrier_ops {
            BarrierOps::Init { barrier, .. } => {
                visit_read(self, barrier);
            }
            BarrierOps::MemCopyAsync {
                barrier,
                source,
                source_length,
                offset_source,
                offset_out,
            } => {
                visit_read(self, barrier);
                visit_read(self, source_length);
                visit_read(self, source);
                visit_read(self, offset_source);
                visit_read(self, offset_out);
            }
            BarrierOps::TmaLoad {
                barrier,
                offset_out,
                tensor_map,
                indices,
            } => {
                visit_read(self, offset_out);
                visit_read(self, barrier);
                visit_read(self, tensor_map);
                for index in indices {
                    visit_read(self, index);
                }
            }
            BarrierOps::TmaLoadIm2col {
                barrier,
                tensor_map,
                indices,
                offset_out,
                offsets,
            } => {
                visit_read(self, offset_out);
                visit_read(self, barrier);
                visit_read(self, tensor_map);
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
            BarrierOps::ExpectTx {
                barrier,
                transaction_count_update,
            } => {
                visit_read(self, barrier);
                visit_read(self, transaction_count_update);
            }
            BarrierOps::Wait { barrier } => {
                visit_read(self, barrier);
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
                offset_source,
            } => {
                visit_read(self, source);
                visit_read(self, offset_source);
                for coord in coordinates {
                    visit_read(self, coord)
                }
            }
            TmaOps::CommitGroup | TmaOps::WaitGroup { .. } | TmaOps::WaitGroupRead { .. } => {}
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

    fn visit_unop(
        &mut self,
        unop: &mut UnaryOperator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        visit_read(self, &mut unop.input);
    }

    fn visit_binop(
        &mut self,
        binop: &mut BinaryOperator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        visit_read(self, &mut binop.lhs);
        visit_read(self, &mut binop.rhs);
    }
}
