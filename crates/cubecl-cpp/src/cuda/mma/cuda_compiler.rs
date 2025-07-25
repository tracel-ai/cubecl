use crate::{
    Dialect,
    cuda::{CudaDialect, arch::CudaArchitecture},
    shared::{
        Architecture, DialectWmmaCompiler, Flags, Fragment, FragmentIdent, FragmentLayout,
        SupportedWmmaCombinations, Variable, WmmaInstruction, variable_to_frag, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};

use super::{MmaSyncCompiler, WMMA_MINIMUM_VERSION, WMMA_NAMESPACE};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

impl DialectWmmaCompiler<CudaDialect<Self>> for CudaWmmaCompiler {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>, _flags: &Flags) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }

    fn compile_wmma_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        flags: &Flags,
    ) -> std::fmt::Result {
        // Only the mma compiler needs to create new type definitions.
        MmaSyncCompiler::<CudaDialect<Self>>::compile_wmma_type_definitions(f, flags)
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        let frag = variable_to_frag(var);

        match is_mma_frag(&frag) {
            true => MmaSyncCompiler::<CudaDialect<Self>>::compile_wmma_fragment_declaration(f, var),
            false => wmma_api_base::compile_fragment_declaration(f, var),
        }
    }

    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        // Only the wmma compiler had ident for the fragment.
        wmma_api_base::compile_fragment_ident(f, WMMA_NAMESPACE, ident)
    }

    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        // Only the wmma compiler can format the layout.
        wmma_api_base::compile_fragment_layout(f, WMMA_NAMESPACE, layout)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match is_mma_frag(fragment) {
            true => MmaSyncCompiler::<CudaDialect<Self>>::compile_wmma_fragment(f, fragment),
            false => wmma_api_base::compile_fragment(f, WMMA_NAMESPACE, fragment),
        }
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match is_mma_inst(instruction) {
            true => MmaSyncCompiler::compile_wmma_instruction(f, instruction),
            false => wmma_api_base::compile_instruction(f, WMMA_NAMESPACE, instruction),
        }
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedWmmaCombinations {
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.get_version() >= WMMA_MINIMUM_VERSION {
            let tdims = vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)];
            // Types fully supported.
            let types = vec![
                (
                    gpu::Elem::Float(gpu::FloatKind::F16), // m
                    gpu::Elem::Float(gpu::FloatKind::F16), // n
                    gpu::Elem::Float(gpu::FloatKind::F16), // k
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::Elem::Int(gpu::IntKind::I8),
                    gpu::Elem::Int(gpu::IntKind::I8),
                    gpu::Elem::Int(gpu::IntKind::I32),
                ),
            ];
            let combinations: SupportedWmmaCombinations = types
                .into_iter()
                .map(|(m, n, k)| {
                    let dimensions = tdims.clone();
                    (m, n, k, dimensions)
                })
                .collect();
            result.extend(combinations);

            if arch.get_version() >= 80 {
                result.push((
                    gpu::Elem::Float(gpu::FloatKind::TF32),
                    gpu::Elem::Float(gpu::FloatKind::TF32),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                    vec![(16, 16, 8)],
                ));
            }

            result.extend(MmaSyncCompiler::<CudaDialect<Self>>::supported_wmma_combinations(arch));
        }
        result
    }
}

fn is_mma_shape(shape: (u8, u8, u8)) -> bool {
    match shape {
        (16, 8, 16) | (8, 8, 32) | (8, 16, 16) => true,
        _ => false,
    }
}
fn is_mma_frag<D: Dialect>(frag: &Fragment<D>) -> bool {
    is_mma_shape((frag.m, frag.n, frag.k))
}

fn is_mma_var<D: Dialect>(var: &Variable<D>) -> bool {
    let frag = variable_to_frag(var);
    is_mma_frag(&frag)
}

fn is_mma_inst<D: Dialect>(instruction: &WmmaInstruction<D>) -> bool {
    match instruction {
        WmmaInstruction::Fill { frag, .. } => is_mma_var(frag),
        WmmaInstruction::Load { frag, .. } => is_mma_var(frag),
        WmmaInstruction::Execute { frag_a, .. } => is_mma_var(frag_a),
        WmmaInstruction::Store { frag, .. } => is_mma_var(frag),
        WmmaInstruction::Cast { input, .. } => is_mma_var(input),
    }
}
