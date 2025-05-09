use crate::{
    hip::{HipDialect, arch::AMDArchitecture},
    shared::{
        DialectWmmaCompiler, Fragment, FragmentIdent, FragmentLayout, SupportedWmmaCombinations,
        WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};

const ROCWMMA_NAMESPACE: &str = "rocwmma";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RocWmmaCompiler {}

impl DialectWmmaCompiler<HipDialect<Self>> for RocWmmaCompiler {
    type Architecture = AMDArchitecture;

    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <rocwmma/rocwmma.hpp>\n")
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_ident(
        ident: &FragmentIdent<HipDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(ROCWMMA_NAMESPACE, ident, f)
    }

    fn compile_fragment_layout(
        layout: &FragmentLayout<HipDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(ROCWMMA_NAMESPACE, layout, f)
    }

    fn compile_fragment(
        fragment: &Fragment<HipDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(ROCWMMA_NAMESPACE, fragment, f)
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<HipDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(ROCWMMA_NAMESPACE, instruction, f)
    }

    fn supported_wmma_combinations(arch: &Self::Architecture) -> SupportedWmmaCombinations {
        match arch {
            Self::Architecture::GFX10 | Self::Architecture::GFX11 => {
                // For gfx11 the supported tile dimensions are always the same
                //                                   m   n   k
                let tdims = vec![(16, 16, 16), (16, 16, 32)];
                let types = vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16), // i
                        gpu::Elem::Float(gpu::FloatKind::F32), // o
                        gpu::Elem::Float(gpu::FloatKind::F32), // c
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                    ),
                ];
                types
                    .into_iter()
                    .map(|(i, o, c)| {
                        let dimensions = tdims.clone();
                        (i, o, c, dimensions)
                    })
                    .collect()
            }
            Self::Architecture::GFX908 => {
                vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F32), // i
                        gpu::Elem::Float(gpu::FloatKind::F32), // o
                        gpu::Elem::Float(gpu::FloatKind::F32), // c
                        vec![
                            //m  n   k
                            (16, 16, 4),
                            (16, 16, 8),
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 2),
                            (32, 32, 4),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 8),
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 4),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 8),
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 4),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        vec![
                            (16, 16, 8),
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 4),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                ]
            }
            Self::Architecture::GFX90A | Self::Architecture::GFX94 => {
                vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F32), // i
                        gpu::Elem::Float(gpu::FloatKind::F32), // o
                        gpu::Elem::Float(gpu::FloatKind::F32), // c
                        vec![
                            //m  n   k
                            (16, 16, 4),
                            (16, 16, 8),
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 2),
                            (32, 32, 4),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        gpu::Elem::Float(gpu::FloatKind::F16),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        gpu::Elem::Float(gpu::FloatKind::BF16),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                ]
            }
            Self::Architecture::Other => vec![],
        }
    }
}
