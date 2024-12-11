use rspirv::{dr::Operand, spirv::Word};

use crate::SpirvCompiler;

use super::{GLCompute, SpirvTarget};

/// To generate:
/// `bindgen GLSL.std.450.h -o GLSL_std_450.rs --default-enum-style rust
#[allow(warnings)]
mod GLSL_std_450;
/// To generate:
/// `bindgen NonSemanticShaderDebugInfo100.h -o NonSemanticShaderDebugInfo100.rs --default-enum-style rust --bitfield-enum .+Flags`
/// grep or equivalent: replace "NonSemanticShaderDebugInfo100" with ""
#[allow(warnings)]
pub mod NonSemanticShaderDebugInfo100;

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
}

pub mod glcompute {
    use super::*;
    use GLSL_std_450::GLSLstd450::{self, *};

    pub const STD_NAME: &str = "GLSL.std.450";

    fn ext_op<T: SpirvTarget, const N: usize>(
        b: &mut SpirvCompiler<T>,
        ty: Word,
        out: Word,
        instruction: GLSLstd450,
        operands: [Word; N],
    ) {
        let ext = b.state.extensions[STD_NAME];
        let operands = operands.into_iter().map(Operand::IdRef).collect::<Vec<_>>();
        b.ext_inst(ty, Some(out), ext, instruction as u32, operands)
            .unwrap();
    }

    impl<T: SpirvTarget> TargetExtensions<T> for GLCompute {
        fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Round, [input]);
        }

        fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450FAbs, [input]);
        }

        fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450SAbs, [input]);
        }

        fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Floor, [input]);
        }

        fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Ceil, [input]);
        }

        fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Sin, [input]);
        }

        fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Cos, [input]);
        }

        fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Tanh, [input]);
        }

        fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Pow, [lhs, rhs]);
        }

        fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Exp, [input]);
        }

        fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Log, [input]);
        }

        fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Sqrt, [input]);
        }

        fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450FMin, [lhs, rhs]);
        }

        fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450UMin, [lhs, rhs]);
        }

        fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450SMin, [lhs, rhs]);
        }

        fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450FMax, [lhs, rhs]);
        }

        fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450UMax, [lhs, rhs]);
        }

        fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450SMax, [lhs, rhs]);
        }

        fn f_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            ext_op(b, ty, out, GLSLstd450FClamp, [input, min, max]);
        }

        fn u_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            ext_op(b, ty, out, GLSLstd450UClamp, [input, min, max]);
        }

        fn s_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            ext_op(b, ty, out, GLSLstd450SClamp, [input, min, max]);
        }

        fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Length, [input]);
        }

        fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            ext_op(b, ty, out, GLSLstd450Normalize, [input]);
        }
    }
}
