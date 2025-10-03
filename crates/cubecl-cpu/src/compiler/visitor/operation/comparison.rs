use cubecl_core::ir::Comparison;
use tracel_llvm::mlir_rs::dialect::arith::{self, CmpfPredicate, CmpiPredicate};

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
            Comparison::IsNan(_op) | Comparison::IsInf(_op) => {
                unreachable!("Should be removed by preprocessor")
            }
        };

        let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
        let (lhs, rhs) = self.visit_correct_index(lhs, rhs);

        let value = if bin_op.lhs.ty.is_float() {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpfPredicate::Olt,
                Comparison::LowerEqual(_) => CmpfPredicate::Ole,
                Comparison::Equal(_) => CmpfPredicate::Oeq,
                Comparison::NotEqual(_) => CmpfPredicate::One,
                Comparison::GreaterEqual(_) => CmpfPredicate::Oge,
                Comparison::Greater(_) => CmpfPredicate::Ogt,
                Comparison::IsNan(_op) | Comparison::IsInf(_op) => unreachable!(),
            };
            self.append_operation_with_result(arith::cmpf(
                self.context,
                predicate,
                lhs,
                rhs,
                self.location,
            ))
        } else if bin_op.lhs.ty.is_signed_int() {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpiPredicate::Slt,
                Comparison::LowerEqual(_) => CmpiPredicate::Sle,
                Comparison::Equal(_) => CmpiPredicate::Eq,
                Comparison::NotEqual(_) => CmpiPredicate::Ne,
                Comparison::GreaterEqual(_) => CmpiPredicate::Sge,
                Comparison::Greater(_) => CmpiPredicate::Sgt,
                Comparison::IsNan(_op) | Comparison::IsInf(_op) => unreachable!(),
            };
            self.append_operation_with_result(arith::cmpi(
                self.context,
                predicate,
                lhs,
                rhs,
                self.location,
            ))
        } else if bin_op.lhs.ty.is_unsigned_int() {
            let predicate = match comparison {
                Comparison::Lower(_) => CmpiPredicate::Ult,
                Comparison::LowerEqual(_) => CmpiPredicate::Ule,
                Comparison::Equal(_) => CmpiPredicate::Eq,
                Comparison::NotEqual(_) => CmpiPredicate::Ne,
                Comparison::GreaterEqual(_) => CmpiPredicate::Uge,
                Comparison::Greater(_) => CmpiPredicate::Ugt,
                Comparison::IsNan(_op) | Comparison::IsInf(_op) => unreachable!(),
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
        let value = self.cast_to_u8(value, out.ty);
        self.insert_variable(out, value);
    }
}
