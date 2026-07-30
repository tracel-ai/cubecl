use cubecl_core as cubecl;
use cubecl_core::ir::dialect::bitwise::{BitwiseNotOp, FindFirstSetOp};
use cubecl_core::ir::dialect::cmp::{FClampOp, SClampOp, UClampOp};
use cubecl_core::ir::dialect::math::{
    ArcCoshOp, ArcSinhOp, ArcTanhOp, DegreesOp, ErfOp, Expm1Op, FModFloorOp, HypotOp, Log1pOp,
    PowiOp, RadiansOp, RecipOp, RhypotOp, RsqrtOp, SModFloorOp, SMulHiOp, SNegOp, UMulHiOp,
};
use cubecl_core::ir::dialect::vector::{FDotOp, MagnitudeOp, NormalizeOp, SDotOp, UDotOp};
use cubecl_core::ir::interfaces::TypedExt;
use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::polyfills::{
    erf, expand_himul_sim, expand_s_himul_64, expand_u_himul_64, expm1, log1p, recip, to_degrees,
    to_radians,
};
use cubecl_core::prelude::*;

use crate::compiler::polyfill::LowerOp;
use cubecl_core::ir::Scope;

macro_rules! lower_unary_math_arith {
    ($cube_op:ty => $polyfill:ident) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let value = self.input(scope.ctx());
                scope.register_value_type::<T, S>(value);
                vec![$polyfill::expand::<T, S>(scope, value.into()).read_value(scope)]
            }
        }
    };
}

macro_rules! lower_binary_math_arith {
    ($cube_op:ty => $polyfill:ident) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let lhs = self.lhs(scope.ctx());
                let rhs = self.rhs(scope.ctx());
                scope.register_value_type::<T, S>(lhs);
                vec![$polyfill::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)]
            }
        }
    };
}

lower_binary_math_arith!(HypotOp => hypot);
lower_binary_math_arith!(RhypotOp => rhypot);

#[cube]
fn dot<T: Numeric, N: Size>(rhs: Vector<T, N>, lhs: Vector<T, N>) -> T {
    (rhs * lhs).vector_sum()
}

lower_binary_math_arith!(FDotOp => dot);
lower_binary_math_arith!(UDotOp => dot);
lower_binary_math_arith!(SDotOp => dot);

#[cube]
pub fn powi<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let one_u = Vector::<i32, N>::new(1);
    let one_t = Vector::<T, N>::new(T::from_int(1));

    let neg = exp.less_than(&Vector::<i32, N>::new(0));
    let mut e = select_many(neg, Vector::<i32, N>::new(0) - exp, exp);

    // TODO: implement leading zero
    let bits = 32; // - u32::leading_zeros(plane_max(Vector::<u32, N>::max_value(e)));

    let mut acc = one_t;
    let mut sq = base;

    for _ in 0..bits {
        acc *= select_many((e & one_u).equal(&one_u), sq, one_t);
        sq *= sq;
        e >>= one_u;
    }

    select_many(neg, one_t / acc, acc)
}

lower_binary_math_arith!(PowiOp => powi);

#[cube]
fn f_mod_floor<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    lhs - rhs * (lhs / rhs).floor()
}

lower_binary_math_arith!(FModFloorOp => f_mod_floor);

#[cube]
fn s_mod_floor<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let zero = Vector::<I, N>::zero();
    let rem = lhs % rhs;
    let signs_differ = rem.less_than(&zero).not_equal(&rhs.less_than(&zero));
    let needs_fixup = rem.not_equal(&zero).vec_and(signs_differ);
    select_many(needs_fixup, rem + rhs, rem)
}

lower_binary_math_arith!(SModFloorOp => s_mod_floor);

#[cube]
fn arc_sinh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    (x + (x * x + Vector::one()).sqrt()).ln()
}

lower_unary_math_arith!(ArcSinhOp => arc_sinh);

#[cube]
fn arc_cosh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    (x + (x * x - Vector::one()).sqrt()).ln()
}

lower_unary_math_arith!(ArcCoshOp => arc_cosh);

#[cube]
fn arc_tanh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    Vector::new(F::new(0.5f32)) * ((Vector::one() + x) / (Vector::one() - x)).ln()
}

lower_unary_math_arith!(ArcTanhOp => arc_tanh);

lower_unary_math_arith!(DegreesOp => to_degrees);
lower_unary_math_arith!(RadiansOp => to_radians);
lower_unary_math_arith!(Log1pOp => log1p);
lower_unary_math_arith!(Expm1Op => expm1);

#[cube]
fn inverse_sqrt<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    Vector::one() / x.sqrt()
}

lower_unary_math_arith!(RsqrtOp => inverse_sqrt);
lower_unary_math_arith!(ErfOp => erf);
lower_unary_math_arith!(RecipOp => recip);

#[cube]
fn neg<I: Int, N: Size>(x: Vector<I, N>) -> Vector<I, N> {
    Vector::zero() - x
}

lower_unary_math_arith!(SNegOp => neg);

#[cube]
fn magnitude<F: Float, N: Size>(x: Vector<F, N>) -> F {
    (x * x).vector_sum().sqrt()
}

lower_unary_math_arith!(MagnitudeOp => magnitude);

#[cube]
fn normalize<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let magnitude = Vector::new((x * x).vector_sum().sqrt());
    x / magnitude
}

lower_unary_math_arith!(NormalizeOp => normalize);

#[cube]
pub fn find_first_set<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    select_many(
        x.equal(&Vector::zero()),
        Vector::zero(),
        x.trailing_zeros() + Vector::one(),
    )
}

lower_unary_math_arith!(FindFirstSetOp => find_first_set);

#[cube]
fn bitwise_not<I: Int, N: Size>(x: Vector<I, N>) -> Vector<I, N> {
    x ^ Vector::from_int(-1)
}

lower_unary_math_arith!(BitwiseNotOp => bitwise_not);

#[op_interface_impl]
impl LowerOp for SMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let lhs = self.lhs(ctx);
        let val = if lhs.size_bits(ctx) == 32 {
            expand_s_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}

#[op_interface_impl]
impl LowerOp for UMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let lhs = self.lhs(ctx);
        let val = if lhs.size_bits(ctx) == 32 {
            expand_u_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}

macro_rules! lower_clamp_math_arith {
    ($cube_op:ty => $polyfill:ident) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let ctx = scope.ctx();
                let input = self.input(ctx);
                let min = self.min(ctx);
                let max = self.max(ctx);
                scope.register_value_type::<T, S>(input);
                let val = $polyfill::expand::<T, S>(scope, input.into(), min.into(), max.into())
                    .read_value(scope);
                vec![val]
            }
        }
    };
}

#[cube]
fn clamp_op<T: Int, N: Size>(
    value: Vector<T, N>,
    min: Vector<T, N>,
    max: Vector<T, N>,
) -> Vector<T, N> {
    value.min(max).max(min)
}

lower_clamp_math_arith!(SClampOp => clamp_op);
lower_clamp_math_arith!(UClampOp => clamp_op);

#[cube]
fn f_clamp_op<T: Float, N: Size>(
    value: Vector<T, N>,
    min: Vector<T, N>,
    max: Vector<T, N>,
) -> Vector<T, N> {
    value.min(max).max(min)
}

lower_clamp_math_arith!(FClampOp => f_clamp_op);
