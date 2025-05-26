use cubecl_core::ir::{Arithmetic, Operation};
use melior::{
    dialect::{arith, ods::vector},
    ir::{BlockLike, Value},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Value<'_, '_>) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            Operation::Arithmetic(Arithmetic::Add(add)) => {
                let result = self
                    .block()
                    .append_operation(arith::addf(
                        self.get_variable_vector(add.lhs),
                        self.get_variable_vector(add.rhs),
                        self.location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();
                self.block().append_operation(
                    vector::store(
                        self.context,
                        result,
                        out,
                        &[self.zero_index()],
                        self.location,
                    )
                    .into(),
                );
            }
            _ => todo!("{} is not implemented yet.", operation),
        }
    }
}
