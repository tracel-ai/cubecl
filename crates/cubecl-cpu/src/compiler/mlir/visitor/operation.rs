use cubecl_core::ir::{
    Arithmetic, Bitwise, Comparison, Elem, IntKind, Metadata, Operation, UIntKind, Variable,
};
use melior::dialect::llvm;
use melior::dialect::ods::llvm as llvm_ods;
use melior::{
    dialect::{
        arith::{self, CmpfPredicate, CmpiPredicate},
        memref,
        ods::vector,
    },
    ir::{
        Attribute, Type, TypeLike, Value, ValueLike, attribute::IntegerAttribute,
        r#type::IntegerType,
    },
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
            Arithmetic::Neg(neg) => {
                let value = self.get_variable(neg.input);
                let result = if neg.input.elem().is_int() {
                    // Cmpl to 2
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
                let result_type = self.item_to_type(unary_operator.input.item);
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type);
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type);
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
                let result_type = self.item_to_type(unary_operator.input.item);
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
                let result_type = self.item_to_type(unary_operator.input.item);
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type).into();
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type).into();
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
                let result_type = self.item_to_type(unary_operator.input.item);
                let value = self.append_operation_with_result(llvm::intr_cttz(
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type).into();
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
                        if let Some(vectorization) = unary_operator.input.item.vectorization {
                            r#type = Type::vector(&[vectorization.get() as u64], r#type).into();
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
