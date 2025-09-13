use rspirv::spirv::Word;

use crate::SpirvCompiler;

use super::{GLCompute, SpirvTarget};

pub trait TargetExtensions<T: SpirvTarget> {
    fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn tan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn cosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn asin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn acos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn asinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn acosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn degrees(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn radians(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atan2(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn f_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn u_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn s_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);

    fn find_msb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn find_lsb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
}

pub mod glcompute {
    use super::*;

    impl<T: SpirvTarget> TargetExtensions<T> for GLCompute {
        fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_round_id(ty, Some(out), input).unwrap();
        }

        fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_f_abs_id(ty, Some(out), input).unwrap();
        }

        fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_s_abs_id(ty, Some(out), input).unwrap();
        }

        fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_floor_id(ty, Some(out), input).unwrap();
        }

        fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_ceil_id(ty, Some(out), input).unwrap();
        }

        fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_sin_id(ty, Some(out), input).unwrap();
        }

        fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_cos_id(ty, Some(out), input).unwrap();
        }

        fn tan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_tan_id(ty, Some(out), input).unwrap();
        }

        fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_tanh_id(ty, Some(out), input).unwrap();
        }

        fn sinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_sinh_id(ty, Some(out), input).unwrap();
        }

        fn cosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_cosh_id(ty, Some(out), input).unwrap();
        }

        fn asin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_asin_id(ty, Some(out), input).unwrap();
        }

        fn acos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_acos_id(ty, Some(out), input).unwrap();
        }

        fn atan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_atan_id(ty, Some(out), input).unwrap();
        }

        fn asinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_asinh_id(ty, Some(out), input).unwrap();
        }

        fn acosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_acosh_id(ty, Some(out), input).unwrap();
        }

        fn atanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_atanh_id(ty, Some(out), input).unwrap();
        }

        fn degrees(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_degrees_id(ty, Some(out), input).unwrap();
        }

        fn radians(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cl_radians_id(ty, Some(out), input).unwrap();
        }

        fn atan2(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.cl_atan2_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_pow_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_exp_id(ty, Some(out), input).unwrap();
        }

        fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_log_id(ty, Some(out), input).unwrap();
        }

        fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_sqrt_id(ty, Some(out), input).unwrap();
        }

        fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_f_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_u_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_s_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_f_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_u_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.gl_s_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn f_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.gl_f_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn u_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.gl_u_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn s_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.gl_s_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_length_id(ty, Some(out), input).unwrap();
        }

        fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_normalize_id(ty, Some(out), input).unwrap();
        }

        fn find_msb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_find_u_msb_id(ty, Some(out), input).unwrap();
        }

        fn find_lsb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.gl_find_i_lsb_id(ty, Some(out), input).unwrap();
        }
    }
}
