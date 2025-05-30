use cubecl_core::ir::{Arithmetic, Comparison, Metadata, Operation, Variable};
use melior::{
    dialect::{
        arith::{self, CmpfPredicate, CmpiPredicate},
        memref,
    },
    ir::{Type, attribute::IntegerAttribute},
};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Variable) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            Operation::Arithmetic(arithmetic) => {
                self.visit_arithmetic(arithmetic, out);
            }
            Operation::Comparison(comparison) => {
                self.visit_comparison(comparison, out);
            }
            Operation::Metadata(metadata) => {
                self.visit_metadata(metadata, out);
            }
            _ => todo!("{:?} is not implemented yet.", operation),
        }
    }

    pub fn visit_metadata(&mut self, metadata: &Metadata, out: Variable) {
        match metadata {
            Metadata::Length { var } => {
                let constant = self.append_operation_with_result(arith::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0).into(),
                    self.location,
                ));
                let variable = self.get_variable(*var);
                let value = self.append_operation_with_result(memref::dim(
                    variable,
                    constant,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            Metadata::BufferLength { var } => {
                let constant = self.append_operation_with_result(arith::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0).into(),
                    self.location,
                ));
                let variable = self.get_variable(*var);
                let value = self.append_operation_with_result(memref::dim(
                    variable,
                    constant,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            _ => todo!("This metadata is not yet implemented {}", metadata),
        }
    }

    pub fn visit_arithmetic(&mut self, arithmetic: &Arithmetic, out: Variable) {
        match arithmetic {
            Arithmetic::Add(add) => {
                let lhs = self.get_variable(add.lhs);
                let rhs = self.get_variable(add.rhs);
                let result =
                    self.append_operation_with_result(arith::addf(lhs, rhs, self.location));
                self.insert_variable(out, result);
            }
            Arithmetic::Dot(_dot) => {
                todo!(
                    "Dot product will needs to be implemented manually, because size are unknown at compilation."
                );
            }
            _ => todo!("This arithmetic is not yet implemented: {}", arithmetic),
        }
    }

    pub fn visit_comparison(&mut self, comparison: &Comparison, out: Variable) {
        let bin_op = match comparison {
            Comparison::Lower(bin_op) => bin_op,
            Comparison::LowerEqual(bin_op) => bin_op,
            Comparison::Equal(bin_op) => bin_op,
            Comparison::NotEqual(bin_op) => bin_op,
            Comparison::GreaterEqual(bin_op) => bin_op,
            Comparison::Greater(bin_op) => bin_op,
        };

        let lhs = self.get_variable(bin_op.lhs);
        let rhs = self.get_variable(bin_op.rhs);
        let (lhs, rhs) = self.visit_correct_index(lhs, rhs);

        let value = if self.is_float(bin_op.lhs.item.elem) {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpfPredicate::Olt,
                Comparison::LowerEqual(_) => CmpfPredicate::Ole,
                Comparison::Equal(_) => CmpfPredicate::Oeq,
                Comparison::NotEqual(_) => CmpfPredicate::One,
                Comparison::GreaterEqual(_) => CmpfPredicate::Oge,
                Comparison::Greater(_) => CmpfPredicate::Ogt,
            };
            self.append_operation_with_result(arith::cmpf(
                self.context,
                predicate,
                lhs,
                rhs,
                self.location,
            ))
        } else if self.is_signed_int(bin_op.lhs.item.elem) {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpiPredicate::Slt,
                Comparison::LowerEqual(_) => CmpiPredicate::Sle,
                Comparison::Equal(_) => CmpiPredicate::Eq,
                Comparison::NotEqual(_) => CmpiPredicate::Ne,
                Comparison::GreaterEqual(_) => CmpiPredicate::Sge,
                Comparison::Greater(_) => CmpiPredicate::Sgt,
            };
            self.append_operation_with_result(arith::cmpi(
                self.context,
                predicate,
                lhs,
                rhs,
                self.location,
            ))
        } else if self.is_unsigned_int(bin_op.lhs.item.elem) {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpiPredicate::Ult,
                Comparison::LowerEqual(_) => CmpiPredicate::Ule,
                Comparison::Equal(_) => CmpiPredicate::Eq,
                Comparison::NotEqual(_) => CmpiPredicate::Ne,
                Comparison::GreaterEqual(_) => CmpiPredicate::Uge,
                Comparison::Greater(_) => CmpiPredicate::Ugt,
            };
            self.append_operation_with_result(arith::cmpi(
                self.context,
                predicate,
                lhs,
                rhs,
                self.location,
            ))
        } else {
            panic!("Impossible comparison");
        };

        self.insert_variable(out, value);
    }
}
