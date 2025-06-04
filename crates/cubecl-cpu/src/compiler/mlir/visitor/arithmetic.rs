use cubecl_core::ir::{Arithmetic, Variable};
use melior::dialect::ods::llvm as llvm_ods;
use melior::ir::BlockLike;
use melior::{
    dialect::{arith, llvm, ods::vector},
    ir::Attribute,
};

use crate::compiler::mlir::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_arithmetic(&mut self, arithmetic: &Arithmetic, out: Variable) {
        match arithmetic {
            Arithmetic::Abs(abs) => {
                let value = self.get_variable(abs.input);
                let result_type = self.item_to_type(abs.input.item);
                let abs = if abs.input.elem().is_int() {
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
                };
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
                let operation = if div.lhs.elem().is_int() {
                    arith::divf(lhs, rhs, self.location)
                } else {
                    arith::divsi(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::Dot(dot) => {
                let lhs = self.get_variable(dot.lhs);
                let rhs = self.get_variable(dot.rhs);
                // This could be used if it wasn't broken and the documentation was usable https://mlir.llvm.org/docs/Dialects/Vector/#vectorcontract-vectorcontractionop
                let result = self.elem_to_type(dot.lhs.elem());
                if dot.lhs.elem().is_int() {
                    let multiplied =
                        self.append_operation_with_result(arith::muli(lhs, rhs, self.location));
                    let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                    let reduction = self.append_operation_with_result(vector::reduction(
                        self.context,
                        result,
                        multiplied,
                        kind,
                        self.location,
                    ));
                    self.insert_variable(out, reduction);
                } else {
                    let multiplied =
                        self.append_operation_with_result(arith::mulf(lhs, rhs, self.location));
                    let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                    let reduction = self.append_operation_with_result(vector::reduction(
                        self.context,
                        result,
                        multiplied,
                        kind,
                        self.location,
                    ));
                    self.insert_variable(out, reduction);
                }
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

                let result_type = self.item_to_type(fma.a.item);
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
                let squared =
                    self.append_operation_with_result(arith::mulf(value, value, self.location));
                let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                let result = self.elem_to_type(magnitude.input.elem());
                let sum = self.append_operation_with_result(vector::reduction(
                    self.context,
                    result,
                    squared,
                    kind,
                    self.location,
                ));
                let squared = self.append_operation_with_result(llvm_ods::intr_sqrt(
                    self.context,
                    sum,
                    self.location,
                ));
                self.insert_variable(out, squared);
            }
            Arithmetic::Max(max) => {
                let lhs = self.get_variable(max.lhs);
                let rhs = self.get_variable(max.rhs);
                let value = if self.is_signed_int(max.lhs.elem()) {
                    self.append_operation_with_result(arith::maxsi(lhs, rhs, self.location))
                } else if self.is_unsigned_int(max.lhs.elem()) {
                    self.append_operation_with_result(arith::maxui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::maxnumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Min(min) => {
                let lhs = self.get_variable(min.lhs);
                let rhs = self.get_variable(min.rhs);
                let value = if self.is_signed_int(min.lhs.elem()) {
                    self.append_operation_with_result(arith::minsi(lhs, rhs, self.location))
                } else if self.is_unsigned_int(min.lhs.elem()) {
                    self.append_operation_with_result(arith::minui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::minimumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Modulo(modulo) => {
                let lhs = self.get_variable(modulo.lhs);
                let rhs = self.get_variable(modulo.rhs);
                let value = if self.is_signed_int(modulo.lhs.elem()) {
                    self.append_operation_with_result(arith::remsi(lhs, rhs, self.location))
                } else if self.is_unsigned_int(modulo.lhs.elem()) {
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
                let operation = if self.is_signed_int(mul_hi.lhs.elem()) {
                    arith::mulsi_extended(lhs, rhs, self.location)
                } else {
                    arith::mului_extended(lhs, rhs, self.location)
                };
                let result = self
                    .block()
                    .append_operation(operation)
                    .result(1)
                    .unwrap()
                    .into();
                self.insert_variable(out, result);
            }
            Arithmetic::Neg(neg) => {
                let value = self.get_variable(neg.input);
                let result = if neg.input.elem().is_int() {
                    // Complement to 2 (inverse + 1)
                    let mask = self.create_int_constant_from_item(neg.input.item, -1);
                    let inv =
                        self.append_operation_with_result(arith::xori(value, mask, self.location)); // Inverse bit
                    let one = self.create_int_constant_from_item(neg.input.item, 1);
                    self.append_operation_with_result(arith::addui_extended(
                        inv,
                        one,
                        self.location,
                    ))
                } else {
                    self.append_operation_with_result(arith::negf(value, self.location))
                };
                self.insert_variable(out, result);
            }
            Arithmetic::Normalize(normalize) => {
                let value = self.get_variable(normalize.input);
                let squared =
                    self.append_operation_with_result(arith::mulf(value, value, self.location));
                let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                let result = self.elem_to_type(normalize.input.elem());
                let sum = self.append_operation_with_result(vector::reduction(
                    self.context,
                    result,
                    squared,
                    kind,
                    self.location,
                ));
                let square_root = self.append_operation_with_result(llvm_ods::intr_sqrt(
                    self.context,
                    sum,
                    self.location,
                ));
                let vector_type = self.item_to_type(normalize.input.item);
                let square_root = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    square_root,
                    self.location,
                ));
                let value = self.append_operation_with_result(arith::divf(
                    value,
                    square_root,
                    self.location,
                ));
                self.insert_variable(out, value);
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
            _ => todo!("This arithmetic is not yet implemented: {}", arithmetic),
        }
    }
}
