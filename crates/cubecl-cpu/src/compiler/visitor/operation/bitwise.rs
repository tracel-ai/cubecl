use cubecl_core::ir::{Bitwise, Elem, IntKind, UIntKind};
use tracel_llvm::melior::{
    dialect::arith::{self, CmpiPredicate},
    dialect::llvm,
    ir::r#type::IntegerType,
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_bitwise(&mut self, bitwise: &Bitwise, out: Variable) {
        let value = match bitwise {
            Bitwise::BitwiseAnd(bin_op) => {
                let (lhs, rhs) = self.get_binary_op_variable(bin_op.lhs, bin_op.rhs);
                self.append_operation_with_result(arith::andi(lhs, rhs, self.location))
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
                let operation = if bin_op.lhs.elem().is_signed_int() {
                    arith::shrsi(lhs, rhs, self.location)
                } else {
                    arith::shrui(lhs, rhs, self.location)
                };
                self.append_operation_with_result(operation)
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
