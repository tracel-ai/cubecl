use cubecl_core::ir::{Arithmetic, Item};
use tracel_llvm::melior::{
    dialect::ods::llvm as llvm_ods,
    dialect::{arith, llvm, ods::vector},
    ir::Attribute,
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_arithmetic(&mut self, arithmetic: &Arithmetic, out: Variable) {
        match arithmetic {
            Arithmetic::Abs(abs) => {
                let value = self.get_variable(abs.input);
                let abs = self.get_absolute_val(abs.input.item, value);
                self.insert_variable(out, abs);
            }
            Arithmetic::Add(add) => {
                let (lhs, rhs) = self.get_binary_op_variable(add.lhs, add.rhs);
                let operation = if add.lhs.elem().is_int() {
                    arith::addi(lhs, rhs, self.location)
                } else {
                    arith::addf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::Ceil(ceil) => {
                let value = self.get_variable(ceil.input);
                let result = self.append_operation_with_result(llvm_ods::intr_ceil(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Clamp(clamp) => {
                let value = self.get_variable(clamp.input);
                let min = self.get_variable(clamp.input);
                let max = self.get_variable(clamp.input);

                let value = if clamp.input.elem().is_signed_int() {
                    let min =
                        self.append_operation_with_result(arith::maxsi(value, min, self.location));
                    self.append_operation_with_result(arith::minsi(min, max, self.location))
                } else if clamp.input.elem().is_unsigned_int() {
                    let min =
                        self.append_operation_with_result(arith::maxui(value, min, self.location));
                    self.append_operation_with_result(arith::minui(min, max, self.location))
                } else {
                    let min = self.append_operation_with_result(arith::maxnumf(
                        value,
                        min,
                        self.location,
                    ));
                    self.append_operation_with_result(arith::minimumf(min, max, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Cos(cos) => {
                let value = self.get_variable(cos.input);
                let result = self.append_operation_with_result(llvm_ods::intr_cos(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Div(div) => {
                let (lhs, rhs) = self.get_binary_op_variable(div.lhs, div.rhs);
                let operation = if div.lhs.elem().is_signed_int() {
                    arith::divsi(lhs, rhs, self.location)
                } else if div.lhs.elem().is_unsigned_int() {
                    arith::divui(lhs, rhs, self.location)
                } else {
                    arith::divf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::Dot(dot) => {
                let lhs = self.get_variable(dot.lhs);
                let rhs = self.get_variable(dot.rhs);
                // This could be used if it wasn't broken and the documentation was usable https://mlir.llvm.org/docs/Dialects/Vector/#vectorcontract-vectorcontractionop
                let result = dot.lhs.elem().to_type(self.context);
                let mul = if dot.lhs.elem().is_int() {
                    arith::muli(lhs, rhs, self.location)
                } else {
                    arith::mulf(lhs, rhs, self.location)
                };
                let mut operation = self.append_operation_with_result(mul);
                if dot.lhs.item.is_vectorized() {
                    let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                    operation = self.append_operation_with_result(vector::reduction(
                        self.context,
                        result,
                        operation,
                        kind,
                        self.location,
                    ));
                }
                self.insert_variable(out, operation);
            }
            Arithmetic::Erf(_) => {
                unreachable!("Should have been transformed in primitive in a previous passe");
            }
            Arithmetic::Exp(exp) => {
                let value = self.get_variable(exp.input);
                let result = self.append_operation_with_result(llvm_ods::intr_exp(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Floor(floor) => {
                let value = self.get_variable(floor.input);
                let result = self.append_operation_with_result(llvm_ods::intr_floor(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Fma(fma) => {
                let a = self.get_variable(fma.a);
                let b = self.get_variable(fma.b);
                let c = self.get_variable(fma.c);

                let result_type = fma.a.item.to_type(self.context);
                let result = self.append_operation_with_result(vector::fma(
                    self.context,
                    result_type,
                    a,
                    b,
                    c,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Log(log) => {
                let value = self.get_variable(log.input);
                let result = self.append_operation_with_result(llvm_ods::intr_log(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Log1p(log) => {
                let value = self.get_variable(log.input);
                let one = self.create_float_constant_from_item(log.input.item, 1.0);
                let plus_one =
                    self.append_operation_with_result(arith::addf(value, one, self.location));
                let result = self.append_operation_with_result(llvm_ods::intr_log(
                    self.context,
                    plus_one,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Magnitude(magnitude) => {
                let value = self.get_variable(magnitude.input);
                let mut squared =
                    self.append_operation_with_result(arith::mulf(value, value, self.location));
                if magnitude.input.item.is_vectorized() {
                    let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                    let result = magnitude.input.elem().to_type(self.context);
                    squared = self.append_operation_with_result(vector::reduction(
                        self.context,
                        result,
                        squared,
                        kind,
                        self.location,
                    ));
                }
                let squared = self.append_operation_with_result(llvm_ods::intr_sqrt(
                    self.context,
                    squared,
                    self.location,
                ));
                self.insert_variable(out, squared);
            }
            Arithmetic::Max(max) => {
                let lhs = self.get_variable(max.lhs);
                let rhs = self.get_variable(max.rhs);
                let value = if max.lhs.elem().is_signed_int() {
                    self.append_operation_with_result(arith::maxsi(lhs, rhs, self.location))
                } else if max.lhs.elem().is_unsigned_int() {
                    self.append_operation_with_result(arith::maxui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::maxnumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Min(min) => {
                let lhs = self.get_variable(min.lhs);
                let rhs = self.get_variable(min.rhs);
                let value = if min.lhs.elem().is_signed_int() {
                    self.append_operation_with_result(arith::minsi(lhs, rhs, self.location))
                } else if min.lhs.elem().is_unsigned_int() {
                    self.append_operation_with_result(arith::minui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::minimumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Modulo(modulo) => {
                let lhs = self.get_variable(modulo.lhs);
                let rhs = self.get_variable(modulo.rhs);
                let value = if modulo.lhs.elem().is_signed_int() {
                    self.append_operation_with_result(arith::remsi(lhs, rhs, self.location))
                } else if modulo.lhs.elem().is_unsigned_int() {
                    self.append_operation_with_result(arith::remui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::remf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Mul(mul) => {
                let (lhs, rhs) = self.get_binary_op_variable(mul.lhs, mul.rhs);
                let operation = if mul.lhs.elem().is_int() {
                    arith::muli(lhs, rhs, self.location)
                } else {
                    arith::mulf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::MulHi(mul_hi) => {
                let (lhs, rhs) = self.get_binary_op_variable(mul_hi.lhs, mul_hi.rhs);
                let operation = if mul_hi.lhs.elem().is_signed_int() {
                    arith::mulsi_extended(lhs, rhs, self.location)
                } else {
                    arith::mului_extended(lhs, rhs, self.location)
                };
                let result = self
                    .block
                    .append_operation(operation)
                    .result(1)
                    .unwrap()
                    .into();
                self.insert_variable(out, result);
            }
            Arithmetic::Neg(neg) => {
                let value = self.get_variable(neg.input);
                let result = if neg.input.elem().is_int() {
                    let zero = self.create_int_constant_from_item(neg.input.item, 0);
                    self.append_operation_with_result(arith::subi(zero, value, self.location))
                } else {
                    self.append_operation_with_result(arith::negf(value, self.location))
                };
                self.insert_variable(out, result);
            }
            Arithmetic::Normalize(normalize) => {
                let value = self.get_variable(normalize.input);
                let result = match normalize.input.item.is_vectorized() {
                    true => {
                        let squared = self.append_operation_with_result(arith::mulf(
                            value,
                            value,
                            self.location,
                        ));
                        let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                        let result = normalize.input.elem().to_type(self.context);
                        let reduced = self.append_operation_with_result(vector::reduction(
                            self.context,
                            result,
                            squared,
                            kind,
                            self.location,
                        ));
                        let square_root = self.append_operation_with_result(llvm_ods::intr_sqrt(
                            self.context,
                            reduced,
                            self.location,
                        ));
                        let vector_type = normalize.input.item.to_type(self.context);
                        let square_root = self.append_operation_with_result(vector::splat(
                            self.context,
                            vector_type,
                            square_root,
                            self.location,
                        ));
                        self.append_operation_with_result(arith::divf(
                            value,
                            square_root,
                            self.location,
                        ))
                    }
                    false => {
                        let abs = self.get_absolute_val(normalize.input.item, value);
                        self.append_operation_with_result(arith::divf(value, abs, self.location))
                    }
                };
                self.insert_variable(out, result);
            }
            Arithmetic::Powf(powf) => {
                let base = self.get_variable(powf.lhs);
                let exp = self.get_variable(powf.rhs);
                let result = self.append_operation_with_result(llvm_ods::intr_pow(
                    self.context,
                    base,
                    exp,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Recip(recip) => {
                let value = self.get_variable(recip.input);
                let one = self.create_float_constant_from_item(recip.input.item, 1.0);
                let recip =
                    self.append_operation_with_result(arith::divf(one, value, self.location));
                self.insert_variable(out, recip);
            }
            Arithmetic::Remainder(remainder) => {
                let lhs = self.get_variable(remainder.lhs);
                let rhs = self.get_variable(remainder.rhs);
                let value = if remainder.lhs.elem().is_signed_int() {
                    self.append_operation_with_result(arith::remsi(lhs, rhs, self.location))
                } else if remainder.lhs.elem().is_unsigned_int() {
                    self.append_operation_with_result(arith::remui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::remf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Round(round) => {
                let input = self.get_variable(round.input);
                let output = self.append_operation_with_result(llvm_ods::intr_round(
                    self.context,
                    input,
                    self.location,
                ));
                self.insert_variable(out, output);
            }
            Arithmetic::Sin(sin) => {
                let input = self.get_variable(sin.input);
                let output = self.append_operation_with_result(llvm_ods::intr_sin(
                    self.context,
                    input,
                    self.location,
                ));
                self.insert_variable(out, output);
            }
            Arithmetic::Sqrt(sqrt) => {
                let input = self.get_variable(sqrt.input);
                let output = self.append_operation_with_result(llvm_ods::intr_sqrt(
                    self.context,
                    input,
                    self.location,
                ));
                self.insert_variable(out, output);
            }
            Arithmetic::Sub(sub) => {
                let (lhs, rhs) = self.get_binary_op_variable(sub.lhs, sub.rhs);
                let operation = if sub.lhs.elem().is_int() {
                    arith::subi(lhs, rhs, self.location)
                } else {
                    arith::subf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::Tanh(tanh) => {
                let input = self.get_variable(tanh.input);
                let output = self.append_operation_with_result(llvm_ods::intr_tanh(
                    self.context,
                    input,
                    self.location,
                ));
                self.insert_variable(out, output);
            }
        }
    }

    fn get_absolute_val(&self, item: Item, value: Value<'a, 'a>) -> Value<'a, 'a> {
        let result_type = item.to_type(self.context);
        if item.elem.is_int() {
            self.append_operation_with_result(llvm::intr_abs(
                self.context,
                value,
                false,
                result_type,
                self.location,
            ))
        } else {
            self.append_operation_with_result(llvm_ods::intr_fabs(
                self.context,
                value,
                self.location,
            ))
        }
    }
}
