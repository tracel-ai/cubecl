use cubecl_core::ir::Comparison;
use tracel_llvm::melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_comparison(&mut self, comparison: &Comparison, out: Variable) {
        let bin_op = match comparison {
            Comparison::Lower(bin_op) => bin_op,
            Comparison::LowerEqual(bin_op) => bin_op,
            Comparison::Equal(bin_op) => bin_op,
            Comparison::NotEqual(bin_op) => bin_op,
            Comparison::GreaterEqual(bin_op) => bin_op,
            Comparison::Greater(bin_op) => bin_op,
        };

        let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
        let (lhs, rhs) = self.visit_correct_index(lhs, rhs);

        let value = if bin_op.lhs.item.elem.is_float() {
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
        } else if bin_op.lhs.item.elem.is_signed_int() {
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
        } else if bin_op.lhs.item.elem.is_unsigned_int() {
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
