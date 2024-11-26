use std::marker::PhantomData;

use crate::shared::{Dialect, IndexedVariable, Variable, WmmaCompiler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HipDialect<M> {
    _wmma_compiler: PhantomData<M>,
}

impl<M: WmmaCompiler<Self>> WmmaCompiler<Self> for HipDialect<M> {
    type Architecture = M::Architecture;

    fn includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::includes(f)
    }

    fn deftypes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::deftypes(f)
    }

    fn local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::local_variables(f)
    }

    fn compile_fragment_ident(
        ident: &crate::shared::FragmentIdent<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        M::compile_fragment_ident(ident, f)
    }

    fn compile_fragment_layout(
        layout: &crate::shared::FragmentLayout<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        M::compile_fragment_layout(layout, f)
    }

    fn compile_fragment(
        fragment: &crate::shared::Fragment<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        M::compile_fragment(fragment, f)
    }

    fn compile_instruction(
        instruction: &crate::shared::WmmaInstruction<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        M::compile_instruction(instruction, f)
    }

    fn supported_wmma_combinations(
        arch: &Self::Architecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        M::supported_wmma_combinations(arch)
    }
}

impl<M: WmmaCompiler<Self>> Dialect for HipDialect<M> {
    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip/hip_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "hip_bf16.h" triggers redefinition errors during compilation
        f.write_str("#include <hip/hip_bfloat16.h>\n")
    }
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip/hip_runtime.h>\n")
    }
    fn include_wmma(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // No need
        Ok(())
    }
    fn bfloat16_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("hip_bfloat16")
    }
    fn bfloat162_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "hip_bfloat16.h" has no "hip_bfloat162" type
        f.write_str("hip_bfloat16")
    }

    fn warp_shuffle(input: &IndexedVariable<Self>, id: &Variable<Self>) -> String {
        format!("__shfl({input}, {id})")
    }
    fn warp_shuffle_xor(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_xor({out}, offset)")
    }
    fn warp_shuffle_down(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_down({out}, offset)")
    }
    fn warp_all(out: &IndexedVariable<Self>) -> String {
        format!("__all({out})")
    }
    fn warp_any(out: &IndexedVariable<Self>) -> String {
        format!("__any({out})")
    }
}
