use cubecl_core::ir::{Instruction, Operation, Operator};
use cubecl_core::{
    ir::{self as core, BinaryOperator, UnaryOperator},
    ExecutionMode,
};
use rspirv::spirv::{Capability, Word};

use crate::{
    item::{Elem, Item},
    lookups::Slice,
    variable::{ConstVal, IndexedVariable},
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, inst: Instruction) {
        match inst.operation {
            Operation::Assign(var) => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(inst.out());
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);

                self.copy_object(ty, Some(out_id), in_id).unwrap();
                self.write(&out, out_id);
            }
            Operation::Operator(operator) => self.compile_operator(operator, inst.out),
            Operation::Atomic(atomic) => self.compile_atomic(atomic, inst.out),
            Operation::Branch(_) => unreachable!("Branches shouldn't exist in optimized IR"),
            Operation::Metadata(meta) => self.compile_meta(meta, inst.out),
            Operation::Subcube(subcube) => self.compile_subcube(subcube, inst.out),
            Operation::Synchronization(sync) => self.compile_sync(sync),
            Operation::CoopMma(cmma) => self.compile_cmma(cmma, inst.out),
        }
    }

    pub fn compile_operator(&mut self, op: Operator, out: Option<core::Variable>) {
        let out = out.unwrap();
        match op {
            Operator::Index(op) => {
                let is_atomic = op.lhs.item.elem.is_atomic();
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);

                if is_atomic {
                    let checked = matches!(self.mode, ExecutionMode::Checked) && value.has_len();
                    let ptr = match self.index(&value, &index, !checked) {
                        IndexedVariable::Pointer(ptr, _) => ptr,
                        _ => unreachable!("Atomic is always pointer"),
                    };
                    let out_id = out.as_binding().unwrap();

                    // This isn't great but atomics can't currently be constructed so should be fine
                    self.merge_binding(out_id, ptr);
                } else {
                    let out_id = self.read_indexed(&out, &value, &index);
                    self.write(&out, out_id);
                }
            }
            Operator::IndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed(&out, &index, value_id);
            }
            Operator::UncheckedIndex(op) => {
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);

                let out_id = self.read_indexed_unchecked(&out, &value, &index);
                self.write(&out, out_id);
            }
            Operator::UncheckedIndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed_unchecked(&out, &index, value_id);
            }
            Operator::Slice(op) => {
                let item = self.compile_item(op.input.item);
                let input = self.compile_variable(op.input);
                let start = self.compile_variable(op.start);
                let end = self.compile_variable(op.end);
                let out = match out.kind {
                    core::VariableKind::Slice { id, depth } => (id, depth),
                    _ => unreachable!(),
                };

                let start_id = self.read(&start);
                let end_id = self.read(&end);
                let const_len = match (start.as_const(), end.as_const()) {
                    (Some(start), Some(end)) => {
                        let len = end.as_u32() - start.as_u32();
                        Some(len)
                    }
                    _ => None,
                };

                self.state.slices.insert(
                    out,
                    Slice {
                        ptr: input,
                        offset: start_id,
                        end: end_id,
                        const_len,
                        item,
                    },
                );
            }
            Operator::Cast(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);

                if let Some(as_const) = input.as_const() {
                    let cast = self.static_cast(as_const, &input.elem(), &out.item());
                    self.copy_object(ty, Some(out_id), cast).unwrap();
                } else {
                    input.item().cast_to(self, Some(out_id), in_id, &out.item());
                }

                self.write(&out, out_id);
            }
            Operator::Equal(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_equal(ty, Some(out), lhs, rhs),
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::NotEqual(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Lower(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::LowerEqual(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than_equal(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Greater(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::GreaterEqual(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Add(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_add(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Sub(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_sub(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_sub(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Mul(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Div(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_div(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Remainder(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mod(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Modulo(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_rem(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_rem(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Dot(op) => {
                if op.lhs.item.vectorization.map(|it| it.get()).unwrap_or(1) == 1 {
                    self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                        match out_ty.elem() {
                            Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                            _ => unreachable!(),
                        };
                    });
                } else {
                    let lhs = self.compile_variable(op.lhs);
                    let rhs = self.compile_variable(op.rhs);
                    let out = self.compile_variable(out);
                    let ty = out.item().id(self);

                    let lhs_id = self.read(&lhs);
                    let rhs_id = self.read(&rhs);
                    let out_id = self.write_id(&out);

                    if matches!(lhs.elem(), Elem::Int(_, _)) {
                        self.capabilities.insert(Capability::DotProduct);
                    }

                    match (lhs.elem(), rhs.elem()) {
                        (Elem::Int(_, false), Elem::Int(_, false)) => {
                            self.u_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Int(_, true), Elem::Int(_, false)) => {
                            self.su_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Int(_, false), Elem::Int(_, true)) => {
                            self.su_dot(ty, Some(out_id), rhs_id, lhs_id, None)
                        }
                        (Elem::Int(_, true), Elem::Int(_, true)) => {
                            self.s_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Float(_), Elem::Float(_)) => {
                            self.dot(ty, Some(out_id), lhs_id, rhs_id)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                    self.write(&out, out_id);
                }
            }
            Operator::Fma(op) => {
                let a = self.compile_variable(op.a);
                let b = self.compile_variable(op.b);
                let c = self.compile_variable(op.c);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let a_id = self.read_as(&a, &out_ty);
                let b_id = self.read_as(&b, &out_ty);
                let c_id = self.read_as(&c, &out_ty);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);

                let mul = self.f_mul(ty, None, a_id, b_id).unwrap();
                self.f_add(ty, Some(out_id), mul, c_id).unwrap();
                self.write(&out, out_id);
            }
            Operator::Recip(op) => {
                self.compile_unary_op_cast(op, out, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    b.f_div(ty, Some(out), one, input).unwrap();
                });
            }
            Operator::And(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.logical_and(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Or(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.logical_or(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Not(op) => {
                self.compile_unary_op_cast(op, out, |b, _, ty, input, out| {
                    b.logical_not(ty, Some(out), input).unwrap();
                });
            }
            Operator::Neg(op) => {
                self.compile_unary_op_cast(op, out, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, true) => b.s_negate(ty, Some(out), input).unwrap(),
                        Elem::Float(_) => b.f_negate(ty, Some(out), input).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::BitwiseAnd(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_and(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Operator::BitwiseOr(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_or(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Operator::BitwiseXor(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_xor(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Operator::ShiftLeft(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.shift_left_logical(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Operator::ShiftRight(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.shift_right_logical(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Operator::Bitcast(op) => self.compile_unary_op(op, out, |b, _, ty, input, out| {
                b.bitcast(ty, Some(out), input).unwrap();
            }),
            Operator::Erf(op) => {
                self.compile_unary_op_cast(op, out, |b, out_ty, ty, input, out| {
                    b.compile_erf(out_ty, ty, input, out);
                })
            }

            // Extension functions
            Operator::Normalize(op) => {
                self.compile_unary_op(op, out, |b, _, ty, input, out| {
                    T::normalize(b, ty, input, out);
                });
            }
            Operator::Magnitude(op) => {
                self.compile_unary_op(op, out, |b, _, ty, input, out| {
                    T::magnitude(b, ty, input, out);
                });
            }
            Operator::Abs(op) => {
                self.compile_unary_op_cast(op, out, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => T::s_abs(b, ty, input, out),
                        Elem::Float(_) => T::f_abs(b, ty, input, out),
                        _ => unreachable!(),
                    }
                });
            }
            Operator::Exp(op) => {
                self.compile_unary_op_cast(op, out, |b, _, ty, input, out| {
                    T::exp(b, ty, input, out)
                });
            }
            Operator::Log(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::log(b, ty, input, out)),
            Operator::Log1p(op) => {
                self.compile_unary_op_cast(op, out, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    let add = match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, None, input, one).unwrap(),
                        Elem::Float(_) => b.f_add(ty, None, input, one).unwrap(),
                        _ => unreachable!(),
                    };
                    T::log(b, ty, add, out)
                });
            }
            Operator::Cos(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::cos(b, ty, input, out)),
            Operator::Sin(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::sin(b, ty, input, out)),
            Operator::Tanh(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::tanh(b, ty, input, out)),
            Operator::Powf(op) => {
                self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                    let bool = match out_ty {
                        Item::Scalar(_) => Elem::Bool.id(b),
                        Item::Vector(_, factor) => Item::Vector(Elem::Bool, factor).id(b),
                        _ => unreachable!(),
                    };
                    let zero = out_ty.const_u32(b, 0);
                    let one = out_ty.const_u32(b, 1);
                    let two = out_ty.const_u32(b, 2);
                    let modulo = b.f_rem(ty, None, rhs, two).unwrap();
                    let is_zero = b.f_ord_equal(bool, None, modulo, zero).unwrap();
                    let abs = b.id();
                    T::f_abs(b, ty, lhs, abs);
                    let even = b.id();
                    T::pow(b, ty, abs, rhs, even);
                    let cond2_0 = b.f_ord_equal(bool, None, modulo, one).unwrap();
                    let cond2_1 = b.f_ord_less_than(bool, None, lhs, zero).unwrap();
                    let cond2 = b.logical_and(bool, None, cond2_0, cond2_1).unwrap();
                    let neg_lhs = b.f_negate(ty, None, lhs).unwrap();
                    let pow2 = b.id();
                    T::pow(b, ty, neg_lhs, rhs, pow2);
                    let pow2_neg = b.f_negate(ty, None, pow2).unwrap();
                    let default = b.id();
                    T::pow(b, ty, lhs, rhs, default);
                    let sel1 = b.select(ty, None, cond2, pow2_neg, default).unwrap();
                    b.select(ty, Some(out), is_zero, even, sel1).unwrap();
                })
            }
            Operator::Sqrt(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::sqrt(b, ty, input, out)),
            Operator::Round(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::round(b, ty, input, out)),
            Operator::Floor(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::floor(b, ty, input, out)),
            Operator::Ceil(op) => self
                .compile_unary_op_cast(op, out, |b, _, ty, input, out| T::ceil(b, ty, input, out)),
            Operator::Clamp(op) => {
                let input = self.compile_variable(op.input);
                let min = self.compile_variable(op.min_value);
                let max = self.compile_variable(op.max_value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let input = self.read_as(&input, &out_ty);
                let min = self.read_as(&min, &out_ty);
                let max = self.read_as(&max, &out_ty);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);

                match out_ty.elem() {
                    Elem::Int(_, false) => T::u_clamp(self, ty, input, min, max, out_id),
                    Elem::Int(_, true) => T::s_clamp(self, ty, input, min, max, out_id),
                    Elem::Float(_) => T::f_clamp(self, ty, input, min, max, out_id),
                    _ => unreachable!(),
                }
                self.write(&out, out_id);
            }

            Operator::Max(op) => self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                match out_ty.elem() {
                    Elem::Int(_, false) => T::u_max(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_max(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_max(b, ty, lhs, rhs, out),
                    _ => unreachable!(),
                }
            }),
            Operator::Min(op) => self.compile_binary_op(op, out, |b, out_ty, ty, lhs, rhs, out| {
                match out_ty.elem() {
                    Elem::Int(_, false) => T::u_min(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_min(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_min(b, ty, lhs, rhs, out),
                    _ => unreachable!(),
                }
            }),

            Operator::InitLine(op) => {
                let values = op
                    .inputs
                    .into_iter()
                    .map(|input| self.compile_variable(input))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|it| self.read(&it))
                    .collect::<Vec<_>>();
                let item = self.compile_item(out.item);
                let out = self.compile_variable(out);
                let out_id = self.write_id(&out);
                let ty = item.id(self);
                self.composite_construct(ty, Some(out_id), values).unwrap();
                self.write(&out, out_id);
            }
            Operator::Copy(op) => {
                let input = self.compile_variable(op.input);
                let in_index = self.compile_variable(op.in_index);
                let out = self.compile_variable(out);
                let out_index = self.compile_variable(op.out_index);

                let in_ptr = self.index_ptr(&input, &in_index);
                let out_ptr = self.index_ptr(&out, &out_index);
                let checked =
                    matches!(self.mode, ExecutionMode::Checked) && input.has_len() && out.has_len();
                if checked {
                    let in_index = self.read(&in_index);
                    let out_index = self.read(&out_index);
                    self.compile_copy_bound(&input, &out, in_index, out_index, None, |b| {
                        b.copy_memory(out_ptr, in_ptr, None, None, vec![]).unwrap();
                    });
                } else {
                    self.copy_memory(out_ptr, in_ptr, None, None, vec![])
                        .unwrap();
                }
            }
            Operator::CopyBulk(op) => {
                self.capabilities.insert(Capability::Addresses);
                let input = self.compile_variable(op.input);
                let in_index = self.compile_variable(op.in_index);
                let out = self.compile_variable(out);
                let out_index = self.compile_variable(op.out_index);

                let source = self.index_ptr(&input, &in_index);
                let target = self.index_ptr(&out, &out_index);
                let size = self.const_u32(op.len * out.item().size());
                let checked =
                    matches!(self.mode, ExecutionMode::Checked) && input.has_len() && out.has_len();
                if checked {
                    let in_index = self.read(&in_index);
                    let out_index = self.read(&out_index);
                    self.compile_copy_bound(&input, &out, in_index, out_index, Some(size), |b| {
                        b.copy_memory_sized(target, source, size, None, None, vec![])
                            .unwrap();
                    });
                } else {
                    self.copy_memory_sized(target, source, size, None, None, vec![])
                        .unwrap();
                }
            }
            Operator::Select(op) => self.compile_select(op.cond, op.then, op.or_else, out),
        }
    }

    pub fn compile_unary_op_cast(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read_as(&input, &out_ty);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op_bool(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let in_ty = input.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);

        let ty = out.item().id(self);

        exec(self, in_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read_as(&lhs, &out_ty);
        let rhs_id = self.read_as(&rhs, &out_ty);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_no_cast(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read(&rhs);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_bool(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let lhs_ty = lhs.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read_as(&rhs, &lhs_ty);
        let out_id = self.write_id(&out);

        let ty = out.item().id(self);

        exec(self, lhs_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    fn compile_erf(&mut self, out_ty: Item, ty: Word, input: Word, out: Word) {
        let bool = match out_ty {
            Item::Scalar(_) => Item::Scalar(Elem::Bool),
            Item::Vector(_, factor) => Item::Vector(Elem::Bool, factor),
            _ => unreachable!(),
        }
        .id(self);
        let mut cast =
            |val: f64| self.static_cast(ConstVal::from_float(val, 64), &Elem::Float(64), &out_ty);
        let p = cast(0.3275911);
        let a1 = cast(0.254829592);
        let a2 = cast(-0.284496736);
        let a3 = cast(1.421413741);
        let a4 = cast(-1.453152027);
        let a5 = cast(1.061405429);
        let one = cast(1.0);
        let zero = cast(0.0);

        let mul = |b: &mut Self, lhs: Word, rhs: Word| b.f_mul(ty, None, lhs, rhs).unwrap();
        let add = |b: &mut Self, lhs: Word, rhs: Word| b.f_add(ty, None, lhs, rhs).unwrap();

        let erf = |b: &mut Self, input: Word| {
            let abs = b.id();
            T::f_abs(b, ty, input, abs);
            let t_0 = mul(b, p, abs);
            let t_1 = add(b, t_0, one);
            let t = b.f_div(ty, None, one, t_1).unwrap();

            let tmp_1 = mul(b, a5, t);
            let tmp_2 = add(b, tmp_1, a4);
            let tmp_3 = mul(b, tmp_2, t);
            let tmp_4 = add(b, tmp_3, a3);
            let tmp_5 = mul(b, tmp_4, t);
            let tmp_6 = add(b, tmp_5, a2);
            let tmp_7 = mul(b, tmp_6, t);
            let tmp = add(b, tmp_7, a1);

            let ret_0 = b.f_negate(ty, None, input).unwrap();
            let ret_1 = mul(b, ret_0, input);
            let ret_2 = b.id();
            T::exp(b, ty, ret_1, ret_2);
            let ret_3 = mul(b, tmp, t);
            let ret_4 = mul(b, ret_2, ret_3);
            b.f_sub(ty, None, one, ret_4).unwrap()
        };

        let cond = self.f_ord_less_than(bool, None, input, zero).unwrap();
        let neg = {
            let neg_in = self.f_negate(ty, None, input).unwrap();
            let res = erf(self, neg_in);
            self.f_negate(ty, None, res).unwrap()
        };
        let pos = erf(self, input);
        self.select(ty, Some(out), cond, neg, pos).unwrap();
    }

    pub fn compile_select(
        &mut self,
        cond: core::Variable,
        then: core::Variable,
        or_else: core::Variable,
        out: core::Variable,
    ) {
        let cond = self.compile_variable(cond);
        let then = self.compile_variable(then);
        let or_else = self.compile_variable(or_else);
        let out = self.compile_variable(out);

        let out_ty = out.item();
        let ty = out_ty.id(self);

        let cond_id = self.read(&cond);
        let then = self.read_as(&then, &out_ty);
        let or_else = self.read_as(&or_else, &out_ty);
        let out_id = self.write_id(&out);

        self.select(ty, Some(out_id), cond_id, then, or_else)
            .unwrap();
        self.write(&out, out_id);
    }
}
