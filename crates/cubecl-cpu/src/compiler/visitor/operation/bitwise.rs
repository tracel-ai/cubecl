use cubecl_core::ir::{Bitwise, ElemType, IntKind, UIntKind};
use tracel_llvm::mlir_rs::{
    dialect::arith::{self, CmpiPredicate},
    dialect::llvm,
    ir::r#type::IntegerType,
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    /// Convert a bit-counting result to u32 output type.
    fn convert_bit_count_to_u32(&mut self, value: Value<'a, 'a>, input: Variable) -> Value<'a, 'a> {
        match input.elem_type() {
            ElemType::Int(IntKind::I8)
            | ElemType::UInt(UIntKind::U8)
            | ElemType::Int(IntKind::I16)
            | ElemType::UInt(UIntKind::U16)
            | ElemType::Int(IntKind::I64)
            | ElemType::UInt(UIntKind::U64) => {
                let mut r#type = IntegerType::new(self.context, 32).into();
                if input.ty.is_vectorized() {
                    r#type = Type::vector(&[input.line_size() as u64], r#type);
                }

                match input.elem_type() {
                    ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64) => self
                        .append_operation_with_result(arith::trunci(value, r#type, self.location)),
                    _ => self.append_operation_with_result(arith::extui(
                        value,
                        r#type,
                        self.location,
                    )),
                }
            }
            ElemType::Int(IntKind::I32) | ElemType::UInt(UIntKind::U32) => value,
            _ => panic!("Unsupported type for bit counting operation"),
        }
    }

    /// Handle leading/trailing zeros with clamping for small types.
    /// For 8/16-bit types, the intrinsic counts based on 8/16 bits but we need to
    /// clamp to the actual bit width since the result could exceed it for zero inputs.
    fn count_zeros_with_clamp(
        &mut self,
        value: Value<'a, 'a>,
        input: Variable,
        out: Variable,
    ) -> Value<'a, 'a> {
        match input.elem_type() {
            ElemType::Int(IntKind::I8)
            | ElemType::UInt(UIntKind::U8)
            | ElemType::Int(IntKind::I16)
            | ElemType::UInt(UIntKind::U16) => {
                let mut r#type = IntegerType::new(self.context, 32).into();
                if input.ty.is_vectorized() {
                    r#type = Type::vector(&[input.line_size() as u64], r#type);
                }
                let value =
                    self.append_operation_with_result(arith::extui(value, r#type, self.location));
                let max = self.create_int_constant_from_item(
                    out.ty,
                    input.ty.storage_type().size_bits() as i64,
                );
                self.append_operation_with_result(arith::minui(value, max, self.location))
            }
            ElemType::Int(IntKind::I32) | ElemType::UInt(UIntKind::U32) => value,
            ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64) => {
                let mut r#type = IntegerType::new(self.context, 32).into();
                if input.ty.is_vectorized() {
                    r#type = Type::vector(&[input.line_size() as u64], r#type);
                }
                self.append_operation_with_result(arith::trunci(value, r#type, self.location))
            }
            _ => panic!("Unsupported type for leading/trailing zeros"),
        }
    }

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
                let operation = if bin_op.lhs.storage_type().is_signed_int() {
                    arith::shrsi(lhs, rhs, self.location)
                } else {
                    arith::shrui(lhs, rhs, self.location)
                };
                self.append_operation_with_result(operation)
            }
            Bitwise::CountOnes(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let result_type = unary_op.input.ty.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_ctpop(
                    value,
                    result_type,
                    self.location,
                ));
                self.convert_bit_count_to_u32(value, unary_op.input)
            }
            Bitwise::ReverseBits(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let result_type = unary_op.input.ty.to_type(self.context);
                self.append_operation_with_result(llvm::intr_bitreverse(
                    value,
                    result_type,
                    self.location,
                ))
            }
            Bitwise::BitwiseNot(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let mask = self.create_int_constant_from_item(unary_op.input.ty, -1);
                self.append_operation_with_result(arith::xori(value, mask, self.location))
            }
            Bitwise::LeadingZeros(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let result_type = unary_op.input.ty.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_ctlz(
                    self.context,
                    value,
                    true,
                    result_type,
                    self.location,
                ));
                self.count_zeros_with_clamp(value, unary_op.input, out)
            }
            Bitwise::TrailingZeros(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let result_type = unary_op.input.ty.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_cttz(
                    self.context,
                    value,
                    true,
                    result_type,
                    self.location,
                ));
                self.count_zeros_with_clamp(value, unary_op.input, out)
            }
            Bitwise::FindFirstSet(unary_op) => {
                let value = self.get_variable(unary_op.input);
                let result_type = unary_op.input.ty.to_type(self.context);
                let value = self.append_operation_with_result(llvm::intr_cttz(
                    self.context,
                    value,
                    false,
                    result_type,
                    self.location,
                ));

                // Add 1 to convert from 0-indexed to 1-indexed
                let one = self.create_int_constant_from_item(unary_op.input.ty, 1);
                let value =
                    self.append_operation_with_result(arith::addi(value, one, self.location));

                // Return 0 if input was 0 (cttz returns bit_width for 0 input)
                let max = self.create_int_constant_from_item(
                    unary_op.input.ty,
                    unary_op.input.ty.storage_type().size_bits() as i64 + 1,
                );
                let cond = self.append_operation_with_result(arith::cmpi(
                    self.context,
                    CmpiPredicate::Uge,
                    value,
                    max,
                    self.location,
                ));
                let zero = self.create_int_constant_from_item(unary_op.input.ty, 0);
                let value = self.append_operation_with_result(arith::select(
                    cond,
                    zero,
                    value,
                    self.location,
                ));

                self.convert_bit_count_to_u32(value, unary_op.input)
            }
        };
        self.insert_variable(out, value);
    }
}
