use cubecl_core::ir::{Operation, Operator};
use cubecl_core::{
    ir::{self as core, BinaryOperator, UnaryOperator},
    ExecutionMode,
};
use rspirv::spirv::{Capability, MemorySemantics, Scope, StorageClass, Word};

use crate::{
    item::{Elem, Item},
    lookups::Slice,
    variable::{ConstVal, IndexedVariable},
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, op: Operation) {
        match op {
            Operation::Operator(operator) => self.compile_operator(operator),
            Operation::Branch(branch) => self.compile_branch(branch),
            Operation::Metadata(meta) => self.compile_meta(meta),
            Operation::Subcube(subcube) => self.compile_subcube(subcube),
            Operation::Procedure(_) => todo!(),
            Operation::Synchronization(sync) => self.compile_sync(sync),
            Operation::CoopMma(cmma) => self.compile_cmma(cmma),
        }
    }

    pub fn compile_operator(&mut self, op: Operator) {
        match op {
            Operator::Index(op) => {
                let is_atomic = op.lhs.item().elem.is_atomic();
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);

                if is_atomic {
                    let checked = matches!(self.mode, ExecutionMode::Checked) && value.has_len();
                    let (ptr, item) = match self.index(&value, &index, !checked) {
                        IndexedVariable::Pointer(ptr, item) => (ptr, item),
                        _ => unreachable!("CMMA store always takes array pointer"),
                    };
                    // This isn't great but atomics can't currently be constructed so should be fine
                    let item = Item::Pointer(StorageClass::StorageBuffer, Box::new(item));
                    let ty = item.id(self);
                    self.copy_object(ty, Some(out_id), ptr).unwrap();
                } else {
                    self.read_indexed(out_id, &value, &index);
                    self.write(&out, out_id);
                }
            }
            Operator::IndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed(&out, &index, value_id);
            }
            Operator::UncheckedIndex(op) => {
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);

                self.read_indexed_unchecked(out_id, &value, &index);
                self.write(&out, out_id);
            }
            Operator::UncheckedIndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed_unchecked(&out, &index, value_id);
            }
            Operator::Slice(op) => {
                let item = self.compile_item(op.input.item());
                let input = self.compile_variable(op.input);
                let start = self.compile_variable(op.start);
                let end = self.compile_variable(op.end);
                let out = match op.out {
                    core::Variable::Slice { id, depth, .. } => (id, depth),
                    _ => unreachable!(),
                };

                let start_id = self.read(&start);
                let (len, const_len) = match (start.as_const(), end.as_const()) {
                    (Some(start), Some(end)) => {
                        let len = end.as_u32() - start.as_u32();
                        let len_id = self.const_u32(len);
                        (len_id, Some(len))
                    }
                    _ => {
                        let end_id = self.read(&end);
                        let len_ty = Elem::Int(32, false).id(self);
                        (self.i_sub(len_ty, None, end_id, start_id).unwrap(), None)
                    }
                };

                self.state.slices.insert(
                    out,
                    Slice {
                        ptr: input,
                        offset: start_id,
                        len,
                        const_len,
                        item,
                    },
                );
            }
            Operator::Assign(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);

                if input.item() == out.item() {
                    self.read_to(&input, out_id);
                } else {
                    let input_id = self.read_as(&input, &out.item());
                    let out_ty = out.item().id(self);
                    self.copy_object(out_ty, Some(out_id), input_id).unwrap();
                };

                self.write(&out, out_id);
            }
            Operator::Equal(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_add(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Sub(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_sub(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_sub(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Mul(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Div(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_div(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Remainder(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_rem(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_rem(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Modulo(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mod(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Dot(op) => {
                if op.lhs.item().vectorization.map(|it| it.get()).unwrap_or(1) == 1 {
                    self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                        match out_ty.elem() {
                            Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                            _ => unreachable!(),
                        };
                    });
                } else {
                    let lhs = self.compile_variable(op.lhs);
                    let rhs = self.compile_variable(op.rhs);
                    let out = self.compile_variable(op.out);
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
                let out = self.compile_variable(op.out);
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
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    b.f_div(ty, Some(out), one, input).unwrap();
                });
            }
            Operator::And(op) => {
                self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                    b.logical_and(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Or(op) => {
                self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                    b.logical_or(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Not(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    b.logical_not(ty, Some(out), input).unwrap();
                });
            }
            Operator::Neg(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, true) => b.s_negate(ty, Some(out), input).unwrap(),
                        Elem::Float(_) => b.f_negate(ty, Some(out), input).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::BitwiseAnd(op) => self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                b.bitwise_and(ty, Some(out), lhs, rhs).unwrap();
            }),
            Operator::BitwiseOr(op) => self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                b.bitwise_or(ty, Some(out), lhs, rhs).unwrap();
            }),
            Operator::BitwiseXor(op) => self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                b.bitwise_xor(ty, Some(out), lhs, rhs).unwrap();
            }),
            Operator::ShiftLeft(op) => self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                b.shift_left_logical(ty, Some(out), lhs, rhs).unwrap();
            }),
            Operator::ShiftRight(op) => self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| {
                b.shift_right_logical(ty, Some(out), lhs, rhs).unwrap();
            }),
            Operator::Bitcast(op) => self.compile_unary_op(op, |b, _, ty, input, out| {
                b.bitcast(ty, Some(out), input).unwrap();
            }),
            Operator::Erf(op) => self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                b.compile_erf(out_ty, ty, input, out);
            }),

            // Extension functions
            Operator::Normalize(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    T::normalize(b, ty, input, out);
                });
            }
            Operator::Magnitude(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    T::magnitude(b, ty, input, out);
                });
            }
            Operator::Abs(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| match out_ty.elem() {
                    Elem::Int(_, _) => T::s_abs(b, ty, input, out),
                    Elem::Float(_) => T::f_abs(b, ty, input, out),
                    _ => unreachable!(),
                });
            }
            Operator::Exp(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::exp(b, ty, input, out));
            }
            Operator::Log(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::log(b, ty, input, out))
            }
            Operator::Log1p(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    let add = match out_ty.elem() {
                        Elem::Int(_, false) => b.i_add(ty, None, input, one).unwrap(),
                        Elem::Float(_) => b.f_add(ty, None, input, one).unwrap(),
                        _ => unreachable!(),
                    };
                    T::exp(b, ty, add, out)
                });
            }
            Operator::Cos(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::cos(b, ty, input, out))
            }
            Operator::Sin(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::sin(b, ty, input, out))
            }
            Operator::Tanh(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::tanh(b, ty, input, out))
            }
            Operator::Powf(op) => {
                self.compile_binary_op(op, |b, _, ty, lhs, rhs, out| T::pow(b, ty, lhs, rhs, out))
            }
            Operator::Sqrt(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::sqrt(b, ty, input, out))
            }
            Operator::Round(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::round(b, ty, input, out))
            }
            Operator::Floor(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::floor(b, ty, input, out))
            }
            Operator::Ceil(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| T::ceil(b, ty, input, out))
            }
            Operator::Clamp(op) => {
                let input = self.compile_variable(op.input);
                let min = self.compile_variable(op.min_value);
                let max = self.compile_variable(op.max_value);
                let out = self.compile_variable(op.out);
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

            Operator::Max(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| match out_ty.elem() {
                    Elem::Int(_, false) => T::u_max(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_max(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_max(b, ty, lhs, rhs, out),
                    _ => unreachable!(),
                })
            }
            Operator::Min(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| match out_ty.elem() {
                    Elem::Int(_, false) => T::u_min(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_min(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_min(b, ty, lhs, rhs, out),
                    _ => unreachable!(),
                })
            }

            // Atomic ops
            Operator::AtomicLoad(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let input_id = input.id();
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_load(ty, Some(out_id), input_id, memory, semantics)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicStore(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(op.out);

                let input_id = self.read(&input);
                let out_id = out.id();

                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_store(out_id, memory, semantics, input_id)
                    .unwrap();
            }
            Operator::AtomicSwap(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_exchange(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicCompareAndSwap(op) => {
                let atomic = self.compile_variable(op.input);
                let cmp = self.compile_variable(op.cmp);
                let val = self.compile_variable(op.val);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let atomic_id = atomic.id();
                let cmp_id = self.read(&cmp);
                let val_id = self.read(&val);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics_success = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());
                let semantics_failure = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_compare_exchange(
                    ty,
                    Some(out_id),
                    atomic_id,
                    memory,
                    semantics_success,
                    semantics_failure,
                    val_id,
                    cmp_id,
                )
                .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicAdd(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_i_add(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicSub(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_i_sub(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicMax(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_max(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_max(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            Operator::AtomicMin(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_min(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_min(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            Operator::AtomicAnd(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_and(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicOr(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_or(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            Operator::AtomicXor(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_ty = out.item();

                let lhs_id = lhs.id();
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_xor(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
        }
    }

    pub fn compile_unary_op(
        &mut self,
        op: UnaryOperator,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
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
            |val: f64| self.static_cast(ConstVal::Bit64(val.to_bits()), &Elem::Float(64), &out_ty);
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
}
