use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::ConstVal,
};
use cubecl_core::ir::{self as core, Arithmetic, InstructionModes};
use rspirv::spirv::{Capability, Decoration, FPEncoding};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_arithmetic(
        &mut self,
        op: Arithmetic,
        out: Option<core::Variable>,
        modes: InstructionModes,
        uniform: bool,
    ) {
        let out = out.unwrap();
        match op {
            Arithmetic::Add(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_add(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            b.f_add(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::SaturatingAdd(_) => {
                unimplemented!("Should be replaced by polyfill");
            }
            Arithmetic::Sub(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_sub(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_sub(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            b.f_sub(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::SaturatingSub(_) => {
                unimplemented!("Should be replaced by polyfill");
            }
            Arithmetic::Mul(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_mul(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            b.f_mul(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::MulHi(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    let out_st = b.type_struct([ty, ty]);
                    let extended = match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mul_extended(out_st, None, lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_mul_extended(out_st, None, lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                    b.composite_extract(ty, Some(out), extended, [1]).unwrap();
                });
            }
            Arithmetic::Div(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_div(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Int(_, true) => {
                            // Convert to float and use `f_mod` (floored division) instead of `s_mod`
                            // (truncated division) to match remainder semantics across dtypes
                            // e.g. remainder(-2, 3) = 1, not 2
                            let f_ty = match out_ty {
                                Item::Scalar(_elem) => Item::Scalar(Elem::Relaxed),
                                Item::Vector(_elem, factor) => Item::Vector(Elem::Relaxed, factor),
                                _ => unreachable!(),
                            };
                            let f_ty = f_ty.id(b);
                            let lhs_f = b.convert_s_to_f(f_ty, None, lhs).unwrap();
                            let rhs_f = b.convert_s_to_f(f_ty, None, rhs).unwrap();
                            let rem = b.f_mod(f_ty, None, lhs_f, rhs_f).unwrap();
                            b.convert_f_to_s(ty, Some(out), rem).unwrap()
                        }
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_mod(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_rem(ty, Some(out), lhs, rhs).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            b.f_rem(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => unreachable!(),
                    };
                });
            }
            Arithmetic::Dot(op) => {
                if op.lhs.ty.line_size() == 1 {
                    self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                        match out_ty.elem() {
                            Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Float(..) => {
                                b.declare_math_mode(modes, out);
                                b.f_mul(ty, Some(out), lhs, rhs).unwrap()
                            }
                            Elem::Relaxed => {
                                b.decorate(out, Decoration::RelaxedPrecision, []);
                                b.declare_math_mode(modes, out);
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
                    if matches!(lhs.elem(), Elem::Float(16, Some(FPEncoding::BFloat16KHR))) {
                        self.capabilities.insert(Capability::BFloat16DotProductKHR);
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
                        (Elem::Float(..), Elem::Float(..))
                        | (Elem::Relaxed, Elem::Float(..))
                        | (Elem::Float(..), Elem::Relaxed) => {
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
                self.declare_math_mode(modes, mul);
                self.f_add(ty, Some(out_id), mul, c_id).unwrap();
                self.declare_math_mode(modes, out_id);
                if relaxed {
                    self.decorate(mul, Decoration::RelaxedPrecision, []);
                    self.decorate(out_id, Decoration::RelaxedPrecision, []);
                }
                self.write(&out, out_id);
            }
            Arithmetic::Recip(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    let one = b.static_cast(ConstVal::Bit32(1), &Elem::Int(32, false), &out_ty);
                    b.declare_math_mode(modes, out);
                    b.f_div(ty, Some(out), one, input).unwrap();
                });
            }
            Arithmetic::Neg(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, true) => b.s_negate(ty, Some(out), input).unwrap(),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_negate(ty, Some(out), input).unwrap()
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                    b.declare_math_mode(modes, out);
                    T::normalize(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                });
            }
            Arithmetic::Magnitude(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            T::f_abs(b, ty, input, out)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            T::f_abs(b, ty, input, out)
                        }
                        _ => unreachable!(),
                    }
                });
            }
            Arithmetic::Exp(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::exp(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                });
            }
            Arithmetic::Log(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
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
                        Elem::Float(..) | Elem::Relaxed => {
                            b.declare_math_mode(modes, out);
                            b.f_add(ty, None, input, one).unwrap()
                        }
                        _ => unreachable!(),
                    };
                    b.mark_uniformity(add, uniform);
                    if relaxed {
                        b.decorate(add, Decoration::RelaxedPrecision, []);
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                    b.declare_math_mode(modes, out);
                    T::log(b, ty, add, out)
                });
            }
            Arithmetic::Cos(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::cos(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Sin(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::sin(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Tan(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::tan(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Tanh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::tanh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Sinh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::sinh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Cosh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::cosh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcCos(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::acos(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcSin(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::asin(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcTan(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::atan(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcSinh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::asinh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcCosh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::acosh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcTanh(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::atanh(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Degrees(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::degrees(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Radians(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::radians(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::ArcTan2(op) => {
                self.compile_binary_op(op, out, uniform, |b, out_ty, ty, lhs, rhs, out| {
                    b.declare_math_mode(modes, out);
                    T::atan2(b, ty, lhs, rhs, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            // No powi for Vulkan, just auto-cast to float
            Arithmetic::Powf(op) | Arithmetic::Powi(op) => {
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
                    b.declare_math_mode(modes, modulo);
                    let is_zero = b.f_ord_equal(bool, None, modulo, zero).unwrap();
                    b.declare_math_mode(modes, is_zero);
                    let abs = b.id();
                    b.declare_math_mode(modes, abs);
                    T::f_abs(b, ty, lhs, abs);
                    let even = b.id();
                    b.declare_math_mode(modes, even);
                    T::pow(b, ty, abs, rhs, even);
                    let cond2_0 = b.f_ord_equal(bool, None, modulo, one).unwrap();
                    b.declare_math_mode(modes, cond2_0);
                    let cond2_1 = b.f_ord_less_than(bool, None, lhs, zero).unwrap();
                    b.declare_math_mode(modes, cond2_1);
                    let cond2 = b.logical_and(bool, None, cond2_0, cond2_1).unwrap();
                    let neg_lhs = b.f_negate(ty, None, lhs).unwrap();
                    b.declare_math_mode(modes, neg_lhs);
                    let pow2 = b.id();
                    b.declare_math_mode(modes, pow2);
                    T::pow(b, ty, neg_lhs, rhs, pow2);
                    let pow2_neg = b.f_negate(ty, None, pow2).unwrap();
                    b.declare_math_mode(modes, pow2_neg);
                    let default = b.id();
                    b.declare_math_mode(modes, default);
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
                    b.declare_math_mode(modes, out);
                    T::sqrt(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::InverseSqrt(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::inverse_sqrt(b, ty, input, out);
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
                    b.declare_math_mode(modes, out);
                    T::floor(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Ceil(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::ceil(b, ty, input, out);
                    if matches!(out_ty.elem(), Elem::Relaxed) {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                    }
                })
            }
            Arithmetic::Trunc(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    b.declare_math_mode(modes, out);
                    T::trunc(b, ty, input, out);
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
                    Elem::Float(..) => {
                        self.declare_math_mode(modes, out_id);
                        T::f_clamp(self, ty, input, min, max, out_id)
                    }
                    Elem::Relaxed => {
                        self.decorate(out_id, Decoration::RelaxedPrecision, []);
                        self.declare_math_mode(modes, out_id);
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
                    Elem::Float(..) => {
                        b.declare_math_mode(modes, out);
                        T::f_max(b, ty, lhs, rhs, out)
                    }
                    Elem::Relaxed => {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                        b.declare_math_mode(modes, out);
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
                    Elem::Float(..) => {
                        b.declare_math_mode(modes, out);
                        T::f_min(b, ty, lhs, rhs, out)
                    }
                    Elem::Relaxed => {
                        b.decorate(out, Decoration::RelaxedPrecision, []);
                        b.declare_math_mode(modes, out);
                        T::f_min(b, ty, lhs, rhs, out)
                    }
                    _ => unreachable!(),
                },
            ),
        }
    }
}
