use cubecl_core::ir::{
    BinaryOperator, CoopMma, Operation, Operator, Select, Subcube, UnaryOperator, Variable,
};

use super::Optimizer;

impl Optimizer {
    pub fn find_writes(&mut self, op: &mut Operation) {
        match op {
            Operation::Operator(operator) => self.find_writes_operator(operator),
            // Metadata is SSA by default since we use once maps
            Operation::Metadata(_) => {}
            // Sync has no outputs
            Operation::Synchronization(_) => {}
            Operation::Subcube(subcube) => self.find_writes_subcube(subcube),
            Operation::CoopMma(coop_mma) => self.find_writes_cmma(coop_mma),
            Operation::Procedure(_) => todo!("Legacy"),
            Operation::Branch(_) => unreachable!(),
        }
    }

    pub fn find_writes_operator(&mut self, op: &mut Operator) {
        match op {
            Operator::Fma(fma_operator) => {
                self.read_var(&mut fma_operator.a);
                self.read_var(&mut fma_operator.b);
                self.read_var(&mut fma_operator.c);
                self.write_var(&mut fma_operator.out)
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
            | Operator::IndexAssign(binary_operator)
            | Operator::UncheckedIndexAssign(binary_operator)
            | Operator::Modulo(binary_operator)
            | Operator::Index(binary_operator)
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
            | Operator::GreaterEqual(binary_operator) => self.visit_binop(binary_operator),

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
            | Operator::Normalize(unary_operator) => self.visit_unop(unary_operator),

            Operator::Clamp(clamp_operator) => {
                self.read_var(&mut clamp_operator.input);
                self.read_var(&mut clamp_operator.min_value);
                self.read_var(&mut clamp_operator.max_value);
                self.write_var(&mut clamp_operator.out);
            }
            Operator::Slice(slice_operator) => {
                self.read_var(&mut slice_operator.start);
                self.read_var(&mut slice_operator.end);
                self.read_var(&mut slice_operator.input);
                self.write_var(&mut slice_operator.out);
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

    fn find_writes_subcube(&mut self, subcube: &mut Subcube) {
        match subcube {
            Subcube::Elect(init_operator) => self.write_var(&mut init_operator.out),
            Subcube::Broadcast(binary_operator) => self.visit_binop(binary_operator),
            Subcube::All(unary_operator)
            | Subcube::Any(unary_operator)
            | Subcube::Sum(unary_operator)
            | Subcube::Prod(unary_operator)
            | Subcube::Min(unary_operator)
            | Subcube::Max(unary_operator) => self.visit_unop(unary_operator),
        }
    }

    fn find_writes_cmma(&mut self, cmma: &mut CoopMma) {
        match cmma {
            CoopMma::Fill { mat, value } => {
                self.read_var(value);
                self.write_var(mat);
            }
            CoopMma::Load { mat, value, stride } => {
                self.read_var(value);
                self.read_var(stride);
                self.write_var(mat);
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => {
                self.read_var(mat_a);
                self.read_var(mat_b);
                self.read_var(mat_c);
                self.write_var(mat_d);
            }
            CoopMma::Store {
                output,
                mat,
                stride,
                ..
            } => {
                self.read_var(mat);
                self.read_var(stride);
                self.write_var(output);
            }
        }
    }

    fn visit_unop(&mut self, unop: &mut UnaryOperator) {
        self.read_var(&mut unop.input);
        self.write_var(&mut unop.out);
    }

    fn visit_binop(&mut self, binop: &mut BinaryOperator) {
        self.read_var(&mut binop.lhs);
        self.read_var(&mut binop.rhs);
        self.write_var(&mut binop.out);
    }

    pub fn read_var(&mut self, var: &mut Variable) {
        if let Some((id, depth)) = self.local_variable_id(var) {
            *var = Variable::LocalBinding {
                id,
                item: var.item(),
                depth,
            }
        }
    }

    pub fn write_var(&mut self, var: &mut Variable) {
        if let Some(id) = self.local_variable_id(var) {
            self.current_block().writes.insert(id);
            self.program.variables.insert(id);
            *var = Variable::LocalBinding {
                id: id.0,
                item: var.item(),
                depth: id.1,
            }
        }
    }

    pub fn find_writes_select(&mut self, select: &mut Select) {
        self.write_var(&mut select.out);
    }
}
