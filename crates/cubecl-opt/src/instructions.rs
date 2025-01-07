use cubecl_core::ir::{
    AtomicOp, BinaryOperator, CoopMma, Instruction, Metadata, Operation, Operator, Plane,
    UnaryOperator, Variable,
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
        self.visit_operation(&mut inst.operation, visit_read);
    }

    /// Visit an operation with a set of read and write visitors. Each visitor will be called with
    /// each read or written to variable.
    pub fn visit_operation(
        &mut self,
        op: &mut Operation,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Operation::Copy(variable) => visit_read(self, variable),
            Operation::Operator(operator) => self.visit_operator(operator, visit_read),
            Operation::Atomic(atomic) => self.visit_atomic(atomic, visit_read),
            Operation::Metadata(meta) => self.visit_meta(meta, visit_read),
            // Sync has no outputs
            Operation::Synchronization(_) | Operation::Debug(_) | Operation::Comment(_) => {}
            Operation::Plane(plane) => self.visit_plane(plane, visit_read),
            Operation::CoopMma(coop_mma) => self.visit_cmma(coop_mma, visit_read),
            Operation::Branch(_) => unreachable!(),
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
            Operator::Fma(fma_operator) => {
                visit_read(self, &mut fma_operator.a);
                visit_read(self, &mut fma_operator.b);
                visit_read(self, &mut fma_operator.c);
            }
            Operator::Add(binary_operator)
            | Operator::Sub(binary_operator)
            | Operator::Mul(binary_operator)
            | Operator::Div(binary_operator)
            | Operator::Powf(binary_operator)
            | Operator::Equal(binary_operator)
            | Operator::NotEqual(binary_operator)
            | Operator::LowerEqual(binary_operator)
            | Operator::UncheckedIndex(binary_operator)
            | Operator::UncheckedIndexAssign(binary_operator)
            | Operator::Modulo(binary_operator)
            | Operator::Index(binary_operator)
            | Operator::IndexAssign(binary_operator)
            | Operator::And(binary_operator)
            | Operator::Greater(binary_operator)
            | Operator::Lower(binary_operator)
            | Operator::Or(binary_operator)
            | Operator::Max(binary_operator)
            | Operator::Min(binary_operator)
            | Operator::BitwiseAnd(binary_operator)
            | Operator::BitwiseOr(binary_operator)
            | Operator::BitwiseXor(binary_operator)
            | Operator::ShiftLeft(binary_operator)
            | Operator::ShiftRight(binary_operator)
            | Operator::Remainder(binary_operator)
            | Operator::Dot(binary_operator)
            | Operator::GreaterEqual(binary_operator) => {
                self.visit_binop(binary_operator, visit_read)
            }

            Operator::Abs(unary_operator)
            | Operator::Exp(unary_operator)
            | Operator::Log(unary_operator)
            | Operator::Log1p(unary_operator)
            | Operator::Cos(unary_operator)
            | Operator::Sin(unary_operator)
            | Operator::Tanh(unary_operator)
            | Operator::Sqrt(unary_operator)
            | Operator::Round(unary_operator)
            | Operator::Floor(unary_operator)
            | Operator::Ceil(unary_operator)
            | Operator::Erf(unary_operator)
            | Operator::Recip(unary_operator)
            | Operator::Not(unary_operator)
            | Operator::Neg(unary_operator)
            | Operator::Cast(unary_operator)
            | Operator::Bitcast(unary_operator)
            | Operator::Magnitude(unary_operator)
            | Operator::Normalize(unary_operator)
            | Operator::CountOnes(unary_operator)
            | Operator::ReverseBits(unary_operator) => self.visit_unop(unary_operator, visit_read),

            Operator::Clamp(clamp_operator) => {
                visit_read(self, &mut clamp_operator.input);
                visit_read(self, &mut clamp_operator.min_value);
                visit_read(self, &mut clamp_operator.max_value);
            }
            Operator::Slice(slice_operator) => {
                visit_read(self, &mut slice_operator.start);
                visit_read(self, &mut slice_operator.end);
                visit_read(self, &mut slice_operator.input);
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
            AtomicOp::Load(unary_operator) | AtomicOp::Store(unary_operator) => {
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
            | Plane::Prod(unary_operator)
            | Plane::Min(unary_operator)
            | Plane::Max(unary_operator) => self.visit_unop(unary_operator, visit_read),
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
            CoopMma::Load { value, stride, .. } => {
                visit_read(self, value);
                visit_read(self, stride);
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
            CoopMma::Store { mat, stride, .. } => {
                visit_read(self, mat);
                visit_read(self, stride);
            }
            CoopMma::Cast { input } => {
                visit_read(self, input);
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
