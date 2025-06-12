use cubecl_core::ir::{
    Bitwise, Comparison, Elem, IntKind, Metadata, Operation, UIntKind, Variable,
};
use tracel_llvm::melior::dialect::{index, llvm};
use tracel_llvm::melior::{
    dialect::{
        arith::{self, CmpfPredicate, CmpiPredicate},
        memref,
    },
    ir::{Type, Value, attribute::IntegerAttribute, r#type::IntegerType},
};

use crate::compiler::mlir::visitor::prelude::IntoType;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation(&mut self, operation: &Operation) {}

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

    pub fn visit_metadata(&mut self, metadata: &Metadata, out: Variable) {
        match metadata {
            Metadata::Length { var } => {
                let constant = self.append_operation_with_result(arith::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0).into(),
                    self.location,
                ));
                let variable = self.get_memory(*var);
                let value = self.append_operation_with_result(memref::dim(
                    variable,
                    constant,
                    self.location,
                ));
                let integer_type = IntegerType::new(self.context, 32);
                let value = self.append_operation_with_result(index::casts(
                    value,
                    integer_type.into(),
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
                let variable = self.get_memory(*var);
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
    pub fn visit_bitwise(&mut self, bitwise: &Bitwise, out: Variable) {
        let value = match bitwise {
            Bitwise::BitwiseAnd(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::addi(lhs, rhs, self.location))
            }
            Bitwise::BitwiseOr(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::ori(lhs, rhs, self.location))
            }
            Bitwise::BitwiseXor(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::xori(lhs, rhs, self.location))
            }
            Bitwise::ShiftLeft(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::shli(lhs, rhs, self.location))
            }
            Bitwise::ShiftRight(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::shrsi(lhs, rhs, self.location))
            }
            Bitwise::CountOnes(unary_operator) => {
                let value = self.get_variable(unary_operator.input);
                let result_type = unary_operator.input.item.to_type(self.context);
                let value: Value<'a, 'a> = self.append_operation_with_result(llvm::intr_ctpop(
                    value,
                    result_type,
                    self.location,
                ));
                match unary_operator.input.item.elem {
                    Elem::Int(IntKind::I8)
                    | Elem::UInt(UIntKind::U8)
                    | Elem::Int(IntKind::I16)
                    | Elem::UInt(UIntKind::U16) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();
                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }

                        self.append_operation_with_result(arith::extui(
                            value,
                            r#type,
                            self.location,
                        ))
                    }
                    Elem::Int(IntKind::I32) | Elem::UInt(UIntKind::U32) => value,
                    Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();
                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }

                        self.append_operation_with_result(arith::trunci(
                            value,
                            r#type,
                            self.location,
                        ))
                    }
                    _ => panic!("These types do not implement count ones"),
                }
            }
            Bitwise::ReverseBits(unary_operator) => {
                let value = self.get_variable(unary_operator.input);
                let result_type = unary_operator.input.item.to_type(self.context);
                self.append_operation_with_result(llvm::intr_bitreverse(
                    value,
                    result_type,
                    self.location,
                ))
            }
            Bitwise::BitwiseNot(unary_operator) => {
                let value = self.get_variable(unary_operator.input);
                let mask = self.create_int_constant_from_item(unary_operator.input.item, -1);
                self.append_operation_with_result(arith::xori(value, mask, self.location))
            }
            Bitwise::LeadingZeros(unary_operator) => {
                let value = self.get_variable(unary_operator.input);
                let result_type = unary_operator.input.item.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_ctlz(
                    self.context,
                    value,
                    true,
                    result_type,
                    self.location,
                ));

                match unary_operator.input.item.elem {
                    Elem::Int(IntKind::I8)
                    | Elem::UInt(UIntKind::U8)
                    | Elem::Int(IntKind::I16)
                    | Elem::UInt(UIntKind::U16) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();

                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }
                        let value = self.append_operation_with_result(arith::extui(
                            value,
                            r#type,
                            self.location,
                        ));
                        let max = self.create_int_constant_from_item(
                            out.item,
                            unary_operator.input.item.elem.size_bits() as i64,
                        );
                        self.append_operation_with_result(arith::minui(value, max, self.location))
                    }
                    Elem::Int(IntKind::I32) | Elem::UInt(UIntKind::U32) => value,
                    Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();
                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }

                        self.append_operation_with_result(arith::trunci(
                            value,
                            r#type,
                            self.location,
                        ))
                    }
                    _ => panic!("These types do not implement count ones"),
                }
            }
            Bitwise::FindFirstSet(unary_operator) => {
                let value = self.get_variable(unary_operator.input);
                let result_type = unary_operator.input.item.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_cttz(
                    self.context,
                    value,
                    false,
                    result_type,
                    self.location,
                ));

                let one = self.create_int_constant_from_item(unary_operator.input.item, 1);
                let value =
                    self.append_operation_with_result(arith::addi(value, one, self.location));

                let max = self.create_int_constant_from_item(
                    unary_operator.input.item,
                    unary_operator.input.item.elem.size_bits() as i64 + 1,
                );
                let cond = self.append_operation_with_result(arith::cmpi(
                    self.context,
                    CmpiPredicate::Uge,
                    value,
                    max,
                    self.location,
                ));
                let zero = self.create_int_constant_from_item(unary_operator.input.item, 0);
                let value = self.append_operation_with_result(arith::select(
                    cond,
                    zero,
                    value,
                    self.location,
                ));

                match unary_operator.input.item.elem {
                    Elem::Int(IntKind::I8)
                    | Elem::UInt(UIntKind::U8)
                    | Elem::Int(IntKind::I16)
                    | Elem::UInt(UIntKind::U16) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();
                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }

                        self.append_operation_with_result(arith::extui(
                            value,
                            r#type,
                            self.location,
                        ))
                    }
                    Elem::Int(IntKind::I32) | Elem::UInt(UIntKind::U32) => value,
                    Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64) => {
                        let mut r#type = IntegerType::new(self.context, 32).into();
                        if unary_operator.input.item.is_vectorized() {
                            r#type = Type::vector(
                                &[unary_operator.input.vectorization_factor() as u64],
                                r#type,
                            );
                        }

                        self.append_operation_with_result(arith::trunci(
                            value,
                            r#type,
                            self.location,
                        ))
                    }
                    _ => panic!("These types do not implement count ones"),
                }
            }
        };
        self.insert_variable(out, value);
    }
}
