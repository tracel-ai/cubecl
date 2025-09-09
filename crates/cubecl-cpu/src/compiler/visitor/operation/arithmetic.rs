use cubecl_core::ir::{self, Arithmetic};
use tracel_llvm::melior::{
    dialect::{
        arith::{self},
        llvm,
        ods::{llvm as llvm_ods, vector},
    },
    ir::Attribute,
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_arithmetic(&mut self, arithmetic: &Arithmetic, out: Variable) {
        match arithmetic {
            Arithmetic::Abs(abs) => {
                let value = self.get_variable(abs.input);
                let abs = self.get_absolute_val(abs.input.ty, value);
                self.insert_variable(out, abs);
            }
            Arithmetic::Add(add) => {
                let (lhs, rhs) = self.get_binary_op_variable(add.lhs, add.rhs);
                let operation = if add.lhs.storage_type().is_int() {
                    arith::addi(lhs, rhs, self.location)
                } else {
                    arith::addf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::ArcCos(acos) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(acos.input);
                let abs = self.get_absolute_val(acos.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(acos.input);
                let result = self.append_operation_with_result(math_ods::acos(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcSin(asin) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(asin.input);
                let abs = self.get_absolute_val(asin.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(asin.input);
                let result = self.append_operation_with_result(math_ods::asin(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcTan(atan) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(atan.input);
                let abs = self.get_absolute_val(atan.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(atan.input);
                let result = self.append_operation_with_result(math_ods::atan(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcSinh(asinh) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(asinh.input);
                let abs = self.get_absolute_val(asinh.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(asinh.input);
                let result = self.append_operation_with_result(math_ods::asinh(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcCosh(acosh) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(acosh.input);
                let abs = self.get_absolute_val(acosh.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(acosh.input);
                let result = self.append_operation_with_result(math_ods::acosh(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcTanh(atanh) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(atanh.input);
                let abs = self.get_absolute_val(atanh.input.ty, value);
                self.insert_variable(out, abs);
                /*let value = self.get_variable(atanh.input);
                let result = self.append_operation_with_result(math_ods::atanh(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);*/
            }
            Arithmetic::ArcTan2(atan2) => {
                // Arc operations are only available through the ods::math module,
                // which can not be properly loaded at the moment.
                // Using dummy for now to satisfy compilation of other tests
                let value = self.get_variable(atan2.lhs);
                let abs = self.get_absolute_val(atan2.lhs.ty, value);
                self.insert_variable(out, abs);
                /*let (lhs, rhs) = self.get_binary_op_variable(atan2.lhs, atan2.rhs);
                let result = self.append_operation_with_result(math_ods::atan_2(
                    self.context,
                    lhs,
                    rhs,
                    self.location,
                ));
                self.insert_variable(out, result);*/
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
                let mut min = self.get_variable(clamp.min_value);
                let mut max = self.get_variable(clamp.max_value);
                let vector_type = Type::vector(
                    &[clamp.input.line_size() as u64],
                    clamp.input.storage_type().to_type(self.context),
                );
                if clamp.input.ty.is_vectorized() && !clamp.min_value.ty.is_vectorized() {
                    min = self.append_operation_with_result(vector::splat(
                        self.context,
                        vector_type,
                        min,
                        self.location,
                    ));
                }

                if clamp.input.ty.is_vectorized() && !clamp.max_value.ty.is_vectorized() {
                    max = self.append_operation_with_result(vector::splat(
                        self.context,
                        vector_type,
                        max,
                        self.location,
                    ));
                }

                let value = if clamp.input.storage_type().is_signed_int() {
                    let clamp_down =
                        self.append_operation_with_result(arith::maxsi(value, min, self.location));
                    self.append_operation_with_result(arith::minsi(clamp_down, max, self.location))
                } else if clamp.input.storage_type().is_unsigned_int() {
                    let clamp_down =
                        self.append_operation_with_result(arith::maxui(value, min, self.location));
                    self.append_operation_with_result(arith::minui(clamp_down, max, self.location))
                } else {
                    let clamp_down = self.append_operation_with_result(arith::maxnumf(
                        value,
                        min,
                        self.location,
                    ));
                    self.append_operation_with_result(arith::minimumf(
                        clamp_down,
                        max,
                        self.location,
                    ))
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
            Arithmetic::Cosh(cosh) => {
                let value = self.get_variable(cosh.input);
                let result = self.append_operation_with_result(llvm_ods::intr_cosh(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Degrees(degrees) => {
                let value = self.get_variable(degrees.input);
                // 180 / pi
                let f = self.create_float_constant_from_item(degrees.input.ty, 57.29577951308232);
                let result =
                    self.append_operation_with_result(arith::mulf(value, f, self.location));
                self.insert_variable(out, result);
            }
            Arithmetic::Div(div) => {
                let (lhs, rhs) = self.get_binary_op_variable(div.lhs, div.rhs);
                let operation = if div.lhs.storage_type().is_signed_int() {
                    arith::divsi(lhs, rhs, self.location)
                } else if div.lhs.storage_type().is_unsigned_int() {
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
                let result = dot.lhs.storage_type().to_type(self.context);
                let mul = if dot.lhs.storage_type().is_int() {
                    arith::muli(lhs, rhs, self.location)
                } else {
                    arith::mulf(lhs, rhs, self.location)
                };
                let mut operation = self.append_operation_with_result(mul);
                if dot.lhs.ty.is_vectorized() {
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

                let result_type = fma.a.ty.to_type(self.context);
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
                let one = self.create_float_constant_from_item(log.input.ty, 1.0);
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
                if magnitude.input.ty.is_vectorized() {
                    let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                    let result = magnitude.input.storage_type().to_type(self.context);
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
                let value = if max.lhs.storage_type().is_signed_int() {
                    self.append_operation_with_result(arith::maxsi(lhs, rhs, self.location))
                } else if max.lhs.storage_type().is_unsigned_int() {
                    self.append_operation_with_result(arith::maxui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::maxnumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Min(min) => {
                let lhs = self.get_variable(min.lhs);
                let rhs = self.get_variable(min.rhs);
                let value = if min.lhs.storage_type().is_signed_int() {
                    self.append_operation_with_result(arith::minsi(lhs, rhs, self.location))
                } else if min.lhs.storage_type().is_unsigned_int() {
                    self.append_operation_with_result(arith::minui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::minimumf(lhs, rhs, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Modulo(modulo) => {
                let (lhs, rhs) = self.get_binary_op_variable(modulo.lhs, modulo.rhs);
                let value = if modulo.lhs.storage_type().is_signed_int() {
                    self.append_operation_with_result(arith::remsi(lhs, rhs, self.location))
                } else if modulo.lhs.storage_type().is_unsigned_int() {
                    self.append_operation_with_result(arith::remui(lhs, rhs, self.location))
                } else {
                    self.append_operation_with_result(arith::remf(lhs, rhs, self.location))
                };

                self.insert_variable(out, value);
            }
            Arithmetic::Mul(mul) => {
                let (lhs, rhs) = self.get_binary_op_variable(mul.lhs, mul.rhs);
                let operation = if mul.lhs.storage_type().is_int() {
                    arith::muli(lhs, rhs, self.location)
                } else {
                    arith::mulf(lhs, rhs, self.location)
                };
                let result = self.append_operation_with_result(operation);
                self.insert_variable(out, result);
            }
            Arithmetic::MulHi(mul_hi) => {
                let (lhs, rhs) = self.get_binary_op_variable(mul_hi.lhs, mul_hi.rhs);
                let operation = if mul_hi.lhs.storage_type().is_signed_int() {
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
                self.insert_variable(out, self.get_neg_val(neg.input));
            }
            Arithmetic::Normalize(normalize) => {
                let value = self.get_variable(normalize.input);
                let result = match normalize.input.ty.is_vectorized() {
                    true => {
                        let squared = self.append_operation_with_result(arith::mulf(
                            value,
                            value,
                            self.location,
                        ));
                        let kind = Attribute::parse(self.context, "#vector.kind<add>").unwrap();
                        let result = normalize.input.storage_type().to_type(self.context);
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
                        let vector_type = normalize.input.ty.to_type(self.context);
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
                        let abs = self.get_absolute_val(normalize.input.ty, value);
                        self.append_operation_with_result(arith::divf(value, abs, self.location))
                    }
                };
                self.insert_variable(out, result);
            }
            Arithmetic::Powf(powf) => {
                let (base, exp) = self.get_binary_op_variable(powf.lhs, powf.rhs);
                let result = self.append_operation_with_result(llvm_ods::intr_pow(
                    self.context,
                    base,
                    exp,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            // Powi intrinsic doesn't accept negative ints for some reason
            Arithmetic::Powi(powi) => {
                let target = powi.lhs.ty.to_type(self.context);
                let (base, exp) = self.get_binary_op_variable(powi.lhs, powi.rhs);
                let exp = self.get_cast_different_type_category(
                    powi.rhs.storage_type(),
                    powi.lhs.storage_type(),
                    target,
                    exp,
                );
                let result = self.append_operation_with_result(llvm_ods::intr_pow(
                    self.context,
                    base,
                    exp,
                    self.location,
                ));
                self.insert_variable(out, result);
            }
            Arithmetic::Radians(radians) => {
                let value = self.get_variable(radians.input);
                // pi / 180
                let f =
                    self.create_float_constant_from_item(radians.input.ty, 0.017453292519943295);
                let result =
                    self.append_operation_with_result(arith::mulf(value, f, self.location));
                self.insert_variable(out, result);
            }
            Arithmetic::Recip(recip) => {
                let value = self.get_variable(recip.input);
                let one = self.create_float_constant_from_item(recip.input.ty, 1.0);
                let recip =
                    self.append_operation_with_result(arith::divf(one, value, self.location));
                self.insert_variable(out, recip);
            }
            Arithmetic::Remainder(remainder) => {
                let (lhs, rhs) = self.get_binary_op_variable(remainder.lhs, remainder.rhs);
                let value = if remainder.lhs.storage_type().is_signed_int() {
                    // TODO: check what is PyTorch behaviour with signed integer
                    self.append_operation_with_result(arith::remsi(lhs, rhs, self.location))
                } else if remainder.lhs.storage_type().is_unsigned_int() {
                    self.append_operation_with_result(arith::remui(lhs, rhs, self.location))
                } else {
                    // To emulate PyTorch behaviour torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
                    let div =
                        self.append_operation_with_result(arith::divf(lhs, rhs, self.location));
                    let floor_div = self.append_operation_with_result(llvm_ods::intr_floor(
                        self.context,
                        div,
                        self.location,
                    ));
                    let coef = self.append_operation_with_result(arith::mulf(
                        floor_div,
                        rhs,
                        self.location,
                    ));

                    self.append_operation_with_result(arith::subf(lhs, coef, self.location))
                };
                self.insert_variable(out, value);
            }
            Arithmetic::Round(round) => {
                let input = self.get_variable(round.input);
                let output = self.append_operation_with_result(llvm_ods::intr_roundeven(
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
            Arithmetic::Sinh(sinh) => {
                let value = self.get_variable(sinh.input);
                let result = self.append_operation_with_result(llvm_ods::intr_sinh(
                    self.context,
                    value,
                    self.location,
                ));
                self.insert_variable(out, result);
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
                let operation = if sub.lhs.storage_type().is_int() {
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

    pub fn get_neg_val(&self, variable: Variable) -> Value<'a, 'a> {
        let value = self.get_variable(variable);
        if variable.storage_type().is_int() {
            let zero = self.create_int_constant_from_item(variable.ty, 0);
            self.append_operation_with_result(arith::subi(zero, value, self.location))
        } else {
            self.append_operation_with_result(arith::negf(value, self.location))
        }
    }

    fn get_absolute_val(&self, item: ir::Type, value: Value<'a, 'a>) -> Value<'a, 'a> {
        let result_type = item.to_type(self.context);
        if item.is_int() {
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
