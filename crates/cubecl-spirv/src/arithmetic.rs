use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::ConstVal,
};
use cubecl_core::ir::{self as core, Arithmetic};
use rspirv::spirv::{Capability, Decoration};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_arithmetic(
        &mut self,
        op: Arithmetic,
        out: Option<core::Variable>,
        uniform: bool,
    ) {
        let out = out.unwrap();
        match op {
            Arithmetic::Add(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_add(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Sub(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_sub(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_sub(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_sub(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Mul(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_mul(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Div(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_div(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Remainder(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_mod(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Modulo(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_rem(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_rem(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_rem(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Dot(op) => {
                if op.lhs.item.vectorization.map(|it| it.get()).unwrap_or(1) == 1 {
                    self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                        match out_ty.elem() {
                            Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Relaxed => {
                                b.decorate(out, Decoration::RelaxedPrecision, []);
                                b.f_mul(ty, Some(out), lhs, rhs).unwrap()
                            }
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
                    self.mark_uniformity(out_id, uniform);

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
                        (Elem::Float(_), Elem::Float(_))
                        | (Elem::Relaxed, Elem::Float(_))
                        | (Elem::Float(_), Elem::Relaxed) => {
                            self.dot(ty, Some(out_id), lhs_id, rhs_id)
                        }
                        (Elem::Relaxed, Elem::Relaxed) => {
                            self.decorate(out_id, Decoration::RelaxedPrecision, []);
                            self.dot(ty, Some(out_id), lhs_id, rhs_id)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                    self.write(&out, out_id);
                }
            }
            Arithmetic::Fma(op) => {
                let a = self.compile_variable(op.a);
                let b = self.compile_variable(op.b);
                let c = self.compile_variable(op.c);
                let out = self.compile_variable(out);
                let out_ty = out.item();
                let relaxed = matches!(
                    (a.item().elem(), b.item().elem(), c.item().elem()),
                    (Elem::Relaxed, Elem::Relaxed, Elem::Relaxed)
                );

                let a_id = self.read_as(&a, &out_ty);
                let b_id = self.read_as(&b, &out_ty);
                let c_id = self.read_as(&c, &out_ty);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);

                let ty = out_ty.id(self);

                let mul = self.f_mul(ty, None, a_id, b_id).unwrap();
                self.mark_uniformity(mul, uniform);
                self.f_add(ty, Some(out_id), mul, c_id).unwrap();
                if relaxed {
                    self.decorate(mul, Decoration::RelaxedPrecision, []);
                    self.decorate(out_id, Decoration::RelaxedPrecision, []);
                }
                self.write(&out, out_id);
            }
            Arithmetic::Recip(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    b.f_div(ty, Some(out), one, input).unwrap();
                });
            }
            Arithmetic::Neg(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, true) => b.s_negate(ty, Some(out), input).unwrap(),
                        Elem::Float(_) => b.f_negate(ty, Some(out), input).unwrap(),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_negate(ty, Some(out), input).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Erf(_) => {
                unreachable!("Replaced by transformer")
            }

            // Extension functions
            Arithmetic::Normalize(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::normalize(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                });
            }
            Arithmetic::Magnitude(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::magnitude(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                });
            }
            Arithmetic::Abs(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => T::s_abs(b, ty, input, out),
                        Elem::Float(_) => T::f_abs(b, ty, input, out),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            T::f_abs(b, ty, input, out)
                        }
                        _ => unreachable!(),
                    }
                });
            }
            Arithmetic::Exp(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::exp(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                });
            }
            Arithmetic::Log(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::log(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Log1p(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    let relaxed = matches!(out_ty.elem(), Elem::Relaxed);
                    let add = match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, None, input, one).unwrap(),
                        Elem::Float(_) | Elem::Relaxed => b.f_add(ty, None, input, one).unwrap(),
                        _ => unreachable!(),
                    };
                    b.mark_uniformity(add, uniform);
                    if relaxed {
                        b.decorate(add, Decoration::RelaxedPrecision, []);
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                    T::log(b, ty, add, out)
                });
            }
            Arithmetic::Cos(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::cos(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Sin(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::sin(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Tanh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::tanh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Powf(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    let bool = match out_ty {
                        Item::Scalar(_) => Elem::Bool.id(b),
                        Item::Vector(_, factor) => Item::Vector(Elem::Bool, factor).id(b),
                        _ => unreachable!(),
                    };
                    let relaxed = matches!(out_ty.elem(), Elem::Relaxed);
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
                    let ids = [
                        modulo, is_zero, abs, even, cond2_0, cond2_1, neg_lhs, pow2, pow2_neg,
                        default,
                    ];
                    for id in ids {
                        b.mark_uniformity(id, uniform);
                        if relaxed {
                            b.decorate(id, Decoration::RelaxedPrecision, []);
                        }
                    }
                    let sel1 = b.select(ty, None, cond2, pow2_neg, default).unwrap();
                    b.mark_uniformity(sel1, uniform);
                    b.select(ty, Some(out), is_zero, even, sel1).unwrap();
                })
            }
            Arithmetic::Sqrt(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::sqrt(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Round(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::round(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Floor(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::floor(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Ceil(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    T::ceil(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Clamp(op) => {
                let input = self.compile_variable(op.input);
                let min = self.compile_variable(op.min_value);
                let max = self.compile_variable(op.max_value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let input = self.read_as(&input, &out_ty);
                let min = self.read_as(&min, &out_ty);
                let max = self.read_as(&max, &out_ty);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);

                let ty = out_ty.id(self);

                match out_ty.elem() {
                    Elem::Int(_, false) => T::u_clamp(self, ty, input, min, max, out_id),
                    Elem::Int(_, true) => T::s_clamp(self, ty, input, min, max, out_id),
                    Elem::Float(_) => T::f_clamp(self, ty, input, min, max, out_id),
                    Elem::Relaxed => {
                        self.decorate(out_id, Decoration::RelaxedPrecision, []);
                        T::f_clamp(self, ty, input, min, max, out_id)
                    }
                    _ => unreachable!(),
                }
                self.write(&out, out_id);
            }

            Arithmetic::Max(op) => self.compile_binary_op(
                op,
                out,
                uniform,
                |b, out_ty, ty, lhs, rhs, out| match out_ty.elem() {
                    Elem::Int(_, false) => T::u_max(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_max(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_max(b, ty, lhs, rhs, out),
                    Elem::Relaxed => {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                        T::f_max(b, ty, lhs, rhs, out)
                    }
                    _ => unreachable!(),
                },
            ),
            Arithmetic::Min(op) => self.compile_binary_op(
                op,
                out,
                uniform,
                |b, out_ty, ty, lhs, rhs, out| match out_ty.elem() {
                    Elem::Int(_, false) => T::u_min(b, ty, lhs, rhs, out),
                    Elem::Int(_, true) => T::s_min(b, ty, lhs, rhs, out),
                    Elem::Float(_) => T::f_min(b, ty, lhs, rhs, out),
                    Elem::Relaxed => {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                        T::f_min(b, ty, lhs, rhs, out)
                    }
                    _ => unreachable!(),
                },
            ),
        }
    }
}
