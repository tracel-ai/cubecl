pub(super) mod arithmetic;
pub(super) mod bitwise;
pub(super) mod comparison;
pub(super) mod metadata;
pub(super) mod operator;

use cubecl_core::ir::Operation;

use crate::compiler::mlir::visitor::prelude::*;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation(&mut self, _operation: &Operation) {}

    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Variable) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            Operation::Arithmetic(arithmetic) => {
                self.visit_arithmetic(arithmetic, out);
            }
            Operation::Bitwise(bitwise) => {
                self.visit_bitwise(bitwise, out);
            }
            Operation::Comparison(comparison) => {
                self.visit_comparison(comparison, out);
            }
            Operation::Metadata(metadata) => {
                self.visit_metadata(metadata, out);
            }
            Operation::Copy(copy) => {
                let value = self.get_variable(*copy);
                self.insert_variable(out, value);
            }
            _ => todo!("{:?} is not implemented yet.", operation),
        }
    }
}
