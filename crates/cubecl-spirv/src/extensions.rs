use rspirv::{dr::Operand, spirv::Word};

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
}

mod glcompute {
    #[allow(non_upper_case_globals)]
    mod ops {
        /// See https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
        use rspirv::spirv::Word;

        pub const Round: Word = 1;
        pub const FAbs: Word = 4;
        pub const SAbs: Word = 5;
        pub const Floor: Word = 8;
        pub const Ceil: Word = 8;
        pub const Sin: Word = 13;
        pub const Cos: Word = 14;
        pub const Tanh: Word = 21;
        pub const Pow: Word = 26;
        pub const Exp: Word = 27;
        pub const Log: Word = 28;
        pub const Sqrt: Word = 31;
        pub const FMin: Word = 37;
        pub const UMin: Word = 38;
        pub const SMin: Word = 39;
        pub const FMax: Word = 40;
        pub const UMax: Word = 41;
        pub const SMax: Word = 42;
        pub const FClamp: Word = 43;
        pub const UClamp: Word = 44;
        pub const SClamp: Word = 45;
        pub const Magnitude: Word = 66;
        pub const Normalize: Word = 69;
    }

    use super::*;
    use ops::*;

    impl<T: SpirvTarget> TargetExtensions<T> for GLCompute {
        fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Round, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, FAbs, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, SAbs, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Floor, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Ceil, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Sin, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Cos, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Tanh, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                Pow,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Exp, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Log, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Sqrt, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                FMin,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                UMin,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                SMin,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                FMax,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                UMax,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                SMax,
                vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
            )
            .unwrap();
        }

        fn f_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                FClamp,
                vec![
                    Operand::IdRef(input),
                    Operand::IdRef(min),
                    Operand::IdRef(max),
                ],
            )
            .unwrap();
        }

        fn u_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                UClamp,
                vec![
                    Operand::IdRef(input),
                    Operand::IdRef(min),
                    Operand::IdRef(max),
                ],
            )
            .unwrap();
        }

        fn s_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            let ext = b.state.extensions[0];
            b.ext_inst(
                ty,
                Some(out),
                ext,
                SClamp,
                vec![
                    Operand::IdRef(input),
                    Operand::IdRef(min),
                    Operand::IdRef(max),
                ],
            )
            .unwrap();
        }

        fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Magnitude, vec![Operand::IdRef(input)])
                .unwrap();
        }

        fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            let ext = b.state.extensions[0];
            b.ext_inst(ty, Some(out), ext, Normalize, vec![Operand::IdRef(input)])
                .unwrap();
        }
    }
}
