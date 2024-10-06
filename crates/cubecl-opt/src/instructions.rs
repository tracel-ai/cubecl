use cubecl_core::ir::{
    BinaryOperator, Branch, CoopMma, Metadata, Operation, Operator, Select, Subcube, UnaryOperator,
    Variable,
};

use super::Optimizer;

impl Optimizer {
    pub fn visit_operation(
        &mut self,
        op: &mut Operation,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Operation::Operator(operator) => self.visit_operator(operator, visit_read, visit_write),
            Operation::Metadata(meta) => self.visit_meta(meta, visit_read, visit_write),
            // Sync has no outputs
            Operation::Synchronization(_) => {}
            Operation::Subcube(subcube) => self.visit_subcube(subcube, visit_read, visit_write),
            Operation::CoopMma(coop_mma) => self.visit_cmma(coop_mma, visit_read, visit_write),
            // Procedures get compiled out before visiting
            Operation::Procedure(_) => {}
            Operation::Branch(Branch::Select(select)) => {
                visit_read(self, &mut select.cond);
                visit_read(self, &mut select.then);
                visit_read(self, &mut select.or_else);
                visit_write(self, &mut select.out);
            }
            Operation::Branch(_) => unreachable!(),
        }
    }

    pub fn visit_operator(
        &mut self,
        op: &mut Operator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        match op {
            Operator::Fma(fma_operator) => {
                visit_read(self, &mut fma_operator.a);
                visit_read(self, &mut fma_operator.b);
                visit_read(self, &mut fma_operator.c);
                visit_write(self, &mut fma_operator.out)
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
                self.visit_binop(binary_operator, visit_read, visit_write)
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
            | Operator::Assign(unary_operator)
            | Operator::Not(unary_operator)
            | Operator::Neg(unary_operator)
            | Operator::Bitcast(unary_operator)
            | Operator::Magnitude(unary_operator)
            | Operator::Normalize(unary_operator) => {
                self.visit_unop(unary_operator, visit_read, visit_write)
            }

            Operator::Clamp(clamp_operator) => {
                visit_read(self, &mut clamp_operator.input);
                visit_read(self, &mut clamp_operator.min_value);
                visit_read(self, &mut clamp_operator.max_value);
                visit_write(self, &mut clamp_operator.out);
            }
            Operator::Slice(slice_operator) => {
                visit_read(self, &mut slice_operator.start);
                visit_read(self, &mut slice_operator.end);
                visit_read(self, &mut slice_operator.input);
                visit_write(self, &mut slice_operator.out);
            }
            Operator::InitLine(line_init_operator) => {
                for input in &mut line_init_operator.inputs {
                    visit_read(self, input)
                }
                visit_write(self, &mut line_init_operator.out)
            }

            // Atomics are always pointers
            Operator::AtomicCompareAndSwap(_)
            | Operator::AtomicLoad(_)
            | Operator::AtomicStore(_)
            | Operator::AtomicSwap(_)
            | Operator::AtomicAdd(_)
            | Operator::AtomicSub(_)
            | Operator::AtomicMax(_)
            | Operator::AtomicMin(_)
            | Operator::AtomicAnd(_)
            | Operator::AtomicOr(_)
            | Operator::AtomicXor(_) => {}
        }
    }

    fn visit_meta(
        &mut self,
        metadata: &mut Metadata,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        match metadata {
            Metadata::Stride { dim, var, out } => {
                visit_read(self, dim);
                visit_read(self, var);
                visit_write(self, out);
            }
            Metadata::Shape { dim, var, out } => {
                visit_read(self, dim);
                visit_read(self, var);
                visit_write(self, out);
            }
            Metadata::Length { var, out } => {
                visit_read(self, var);
                visit_write(self, out);
            }
        }
    }

    fn visit_subcube(
        &mut self,
        subcube: &mut Subcube,
        visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        match subcube {
            Subcube::Elect(init_operator) => visit_write(self, &mut init_operator.out),
            Subcube::Broadcast(binary_operator) => {
                self.visit_binop(binary_operator, visit_read, visit_write)
            }
            Subcube::All(unary_operator)
            | Subcube::Any(unary_operator)
            | Subcube::Sum(unary_operator)
            | Subcube::Prod(unary_operator)
            | Subcube::Min(unary_operator)
            | Subcube::Max(unary_operator) => {
                self.visit_unop(unary_operator, visit_read, visit_write)
            }
        }
    }

    fn visit_cmma(
        &mut self,
        cmma: &mut CoopMma,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        match cmma {
            CoopMma::Fill { mat, value } => {
                visit_read(self, value);
                visit_write(self, mat);
            }
            CoopMma::Load { mat, value, stride } => {
                visit_read(self, value);
                visit_read(self, stride);
                visit_write(self, mat);
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => {
                visit_read(self, mat_a);
                visit_read(self, mat_b);
                visit_read(self, mat_c);
                visit_write(self, mat_d);
            }
            CoopMma::Store {
                output,
                mat,
                stride,
                ..
            } => {
                visit_read(self, mat);
                visit_read(self, stride);
                visit_write(self, output);
            }
        }
    }

    fn visit_unop(
        &mut self,
        unop: &mut UnaryOperator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        visit_read(self, &mut unop.input);
        visit_write(self, &mut unop.out);
    }

    fn visit_binop(
        &mut self,
        binop: &mut BinaryOperator,
        mut visit_read: impl FnMut(&mut Self, &mut Variable),
        mut visit_write: impl FnMut(&mut Self, &mut Variable),
    ) {
        visit_read(self, &mut binop.lhs);
        visit_read(self, &mut binop.rhs);
        visit_write(self, &mut binop.out);
    }

    pub fn write_var(&mut self, var: &mut Variable) {
        if let Some(id) = self.local_variable_id(var) {
            self.current_block_mut().writes.insert(id);
        }
    }

    pub fn find_writes_select(&mut self, select: &mut Select) {
        self.write_var(&mut select.out);
    }
}
