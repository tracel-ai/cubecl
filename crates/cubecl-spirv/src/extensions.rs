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
    fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
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
    use rspirv_ext::dr::autogen_glsl_std_450::GLOpBuilder;

    use super::*;

    impl<T: SpirvTarget> TargetExtensions<T> for GLCompute {
        fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.round_id(ty, Some(out), input).unwrap();
        }

        fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.f_abs_id(ty, Some(out), input).unwrap();
        }

        fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.s_abs_id(ty, Some(out), input).unwrap();
        }

        fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.floor_id(ty, Some(out), input).unwrap();
        }

        fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.ceil_id(ty, Some(out), input).unwrap();
        }

        fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.sin_id(ty, Some(out), input).unwrap();
        }

        fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.cos_id(ty, Some(out), input).unwrap();
        }

        fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.tanh_id(ty, Some(out), input).unwrap();
        }

        fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.pow_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.exp_id(ty, Some(out), input).unwrap();
        }

        fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.log_id(ty, Some(out), input).unwrap();
        }

        fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.sqrt_id(ty, Some(out), input).unwrap();
        }

        fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.f_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.u_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.s_min_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.f_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.u_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            b.s_max_id(ty, Some(out), lhs, rhs).unwrap();
        }

        fn f_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.f_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn u_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.u_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn s_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            b.s_clamp_id(ty, Some(out), input, min, max).unwrap();
        }

        fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.length_id(ty, Some(out), input).unwrap();
        }

        fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.normalize_id(ty, Some(out), input).unwrap();
        }

        fn find_msb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.find_u_msb_id(ty, Some(out), input).unwrap();
        }

        fn find_lsb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            b.find_i_lsb_id(ty, Some(out), input).unwrap();
        }
    }
}
