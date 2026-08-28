use alloc::vec;
use core::f32::consts::PI;

use cubecl_ir::{Type, cube_op, prelude::*};
use num_traits::One;

use crate::prelude::*;
use crate::{self as cubecl, unexpanded};

define_scalar!(ElemA);
define_size!(SizeA);

/// Change the meaning of the given cube primitive type during compilation.
///
/// # Warning
///
/// To be used for very custom kernels, it would likely lead to a JIT compiler error otherwise.
pub fn set_polyfill<E: Scalar, N: Size>(_elem: Type) {
    unexpanded!()
}

/// Expand module of [`set_polyfill()`].
pub mod set_polyfill {
    use super::*;

    /// Expand function of [`set_polyfill()`].
    pub fn expand<E: Scalar, N: Size>(scope: &Scope, ty: Type) {
        scope.register_type::<E>(ty.elem_type());
        scope.register_size::<N>(ty.vector_size());
    }
}

#[cube]
pub fn erf<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let erf = erf_positive(x.abs());
    select_many(x.less_than(&Vector::new(F::new(0f32))), -erf, erf)
}

/// An approximation of the error function: <https://en.wikipedia.org/wiki/Error_function#Numerical_approximations>
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
#[cube]
fn erf_positive<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let p = Vector::new(F::new(0.3275911_f32));
    let a1 = Vector::new(F::new(0.2548296_f32));
    let a2 = Vector::new(F::new(-0.28449674_f32));
    let a3 = Vector::new(F::new(1.4214137_f32));
    let a4 = Vector::new(F::new(-1.453152_f32));
    let a5 = Vector::new(F::new(1.0614054_f32));
    let one = Vector::new(F::new(1.0_f32));

    let t = one / (one + p * x);
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    one - (tmp * t * (-x * x).exp())
}

#[cube]
fn himul_i64<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<i64, N>::cast_from(lhs) * Vector::<i64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[cube]
fn himul_u64<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<u64, N>::cast_from(lhs) * Vector::<u64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[allow(missing_docs)]
pub fn expand_s_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_i64::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[allow(missing_docs)]
pub fn expand_u_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_u64::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[cube]
fn himul_sim<T: Int, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
    let half_bits = T::size_bits().comptime() / 2;
    let low_mask = Vector::new(T::new(comptime!((1i64 << half_bits) - 1)));
    let shift = Vector::new(T::new(half_bits as i64));

    let lhs_low = lhs & low_mask;
    let lhs_hi = (lhs >> shift) & low_mask;
    let rhs_low = rhs & low_mask;
    let rhs_hi = (rhs >> shift) & low_mask;

    let low_low = lhs_low * rhs_low;
    let high_low = lhs_hi * rhs_low;
    let low_high = lhs_low * rhs_hi;
    let high_high = lhs_hi * rhs_hi;

    let mid = ((low_low >> shift) & low_mask) + (high_low & low_mask) + (low_high & low_mask);
    high_high
        + ((high_low >> shift) & low_mask)
        + ((low_high >> shift) & low_mask)
        + ((mid >> shift) & low_mask)
}

#[allow(missing_docs)]
pub fn expand_himul_sim(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_sim::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[cube]
pub fn log1p<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    (input + Vector::one()).ln()
}

#[cube]
pub fn expm1<T: Float, N: Size>(x: Vector<T, N>) -> Vector<T, N> {
    let sq = x * x;
    let a = sq * Vector::new(T::new(0.5_f32));
    let b = sq * x * Vector::new(T::new(1.0_f32 / 6.0_f32));
    let taylor = x + a + b;
    let is_small = x.abs().less_than(&Vector::new(T::new(1e-5_f32)));
    select_many(is_small, taylor, x.exp() - Vector::one())
}

/// `powf` without any edge case handling. Useful as a common mapping for the backend version that
/// doesn't handle edge cases normally.
#[cube_op(name = "polyfill.simple_pow")]
#[result_ty(same_as = base)]
pub struct SimplePowOp {
    pub base: Value,
    pub exp: Value,
}

/// use the simple version because otherwise we'd get an infinite lowering loop
#[cube]
fn simple_pow<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let base = base.read_value(scope);
        let exp = exp.read_value(scope);
        let powf = SimplePowOp::new(scope.ctx_mut(), base, exp);
        scope.register_with_result(&powf).into()
    })
}

#[cube]
pub fn powf<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    let modulo = exp.mod_floor(Vector::new(T::new(2.0_f32)));
    let is_even = modulo.equal(&Vector::zero());
    let is_odd = modulo.equal(&Vector::one());
    let is_neg_base = base.less_than(&Vector::zero());

    let even_res = simple_pow(base.abs(), exp);
    let odd_neg_res = -(simple_pow(-base, exp));
    let default = simple_pow(base, exp);

    let sel1 = select_many(is_odd.vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

#[cube]
pub fn powi<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let is_even = exp.is_multiple_of(2);
    let is_neg_base = base.less_than(&Vector::zero());
    let exp = Vector::cast_from(exp);

    let even_res = simple_pow(base.abs(), exp);
    let odd_neg_res = -(simple_pow(-base, exp));
    let default = simple_pow(base, exp);

    let sel1 = select_many((!is_even).vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

#[cube]
pub fn recip<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::one() / input
}

#[cube]
pub fn to_degrees<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(180.0_f32 / PI)))
}

#[cube]
pub fn to_radians<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(PI / 180.0_f32)))
}

pub mod bitwise {
    use super::*;

    #[cube]
    pub fn u64_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_zeros = Vector::leading_zeros(low);
        let high_zeros = Vector::leading_zeros(high);

        select_many(
            high_zeros.equal(&Vector::new(32)),
            low_zeros + high_zeros,
            high_zeros,
        )
    }

    #[cube]
    pub fn u64_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_tz = Vector::trailing_zeros(low);
        let high_tz = Vector::trailing_zeros(high);

        let high_tz = select_many(
            high_tz.equal(&Vector::new(32)),
            Vector::new(64),
            high_tz + Vector::new(32),
        );
        select_many(low_tz.equal(&Vector::new(32)), high_tz, low_tz)
    }

    #[cube]
    pub fn u64_ffs<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_ffs = Vector::find_first_set(low);
        let high_ffs = Vector::find_first_set(high);

        let high_ffs = select_many(
            high_ffs.equal(&Vector::new(0)),
            high_ffs,
            high_ffs + Vector::new(32),
        );
        select_many(low_ffs.equal(&Vector::new(0)), high_ffs, low_ffs)
    }
}

/// The plane reductions and scans, as folds over the shuffles.
///
/// A backend that has cross-lane shuffles but no reduction of its own gets them from here; the
/// C++ backends and the LLVM one share these.
pub mod plane {
    use super::*;
    use crate::prelude::{
        CUBE_DIM, CubeAdd, CubeMul, CubePartialOrd, PLANE_DIM, UNIT_POS_PLANE, max, min,
        plane_shuffle_up, plane_shuffle_xor, select,
    };

    #[cube]
    pub trait PlaneOp<T: Scalar, N: Size> {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N>;
    }

    pub struct OpAdd;
    pub struct OpMul;
    pub struct OpMin;
    pub struct OpMax;

    #[cube]
    impl<T: Scalar + CubeAdd, N: Size> PlaneOp<T, N> for OpAdd {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            lhs + rhs
        }
    }
    #[cube]
    impl<T: Scalar + CubeMul, N: Size> PlaneOp<T, N> for OpMul {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            lhs * rhs
        }
    }
    #[cube]
    impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMin {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            min(lhs, rhs)
        }
    }
    #[cube]
    impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMax {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            max(lhs, rhs)
        }
    }

    #[cube]
    fn plane_dim_checked() -> u32 {
        min(PLANE_DIM, CUBE_DIM)
    }

    #[cube]
    pub fn plane_reduce<T: Scalar, N: Size, Op: PlaneOp<T, N>>(val: Vector<T, N>) -> Vector<T, N> {
        let plane_dim = plane_dim_checked();
        let mut acc = val;
        let mut offset = 1;
        while offset < plane_dim {
            acc = Op::apply(acc, plane_shuffle_xor(acc, offset));
            offset *= 2;
        }
        acc
    }

    #[cube]
    pub fn plane_reduce_inclusive<T: Scalar, N: Size, Op: PlaneOp<T, N>>(
        val: Vector<T, N>,
    ) -> Vector<T, N> {
        let plane_dim = plane_dim_checked();
        let mut acc = val;
        let mut offset = 1;
        while offset < plane_dim {
            let tmp = Op::apply(acc, plane_shuffle_up(acc, offset));
            if UNIT_POS_PLANE >= offset {
                acc = tmp;
            }
            offset *= 2;
        }
        acc
    }

    #[cube]
    pub fn plane_reduce_exclusive<T: Numeric, N: Size, Op: PlaneOp<T, N>>(
        val: Vector<T, N>,
        #[comptime] default: i64,
    ) -> Vector<T, N> {
        let inclusive = plane_reduce_inclusive::<T, N, Op>(val);
        let shfl = plane_shuffle_up(inclusive, 1);
        select(UNIT_POS_PLANE == 0, Vector::new(T::from_int(default)), shfl)
    }
}
