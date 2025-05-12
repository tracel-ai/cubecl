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
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <rocwmma/rocwmma.hpp>\n")
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(f, ROCWMMA_NAMESPACE, ident)
    }

    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(f, ROCWMMA_NAMESPACE, layout)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(f, ROCWMMA_NAMESPACE, fragment)
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(f, ROCWMMA_NAMESPACE, instruction)
    }

    fn supported_wmma_combinations(arch: &AMDArchitecture) -> SupportedWmmaCombinations {
        match arch {
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 => {
                // For gfx11 the supported tile dimensions are always the same
                //                                   m   n   k
                let tdims = vec![(16, 16, 16), (16, 16, 32)];
                let types = vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F16), // m / i
                        gpu::Elem::Float(gpu::FloatKind::F32), // n / o
                        gpu::Elem::Float(gpu::FloatKind::F32), // k / c
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
            AMDArchitecture::GFX908 => {
                vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F32), // m / i
                        gpu::Elem::Float(gpu::FloatKind::F32), // n / o
                        gpu::Elem::Float(gpu::FloatKind::F32), // k / c
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
            AMDArchitecture::GFX90A | AMDArchitecture::GFX94 => {
                vec![
                    (
                        gpu::Elem::Float(gpu::FloatKind::F32), // m / i
                        gpu::Elem::Float(gpu::FloatKind::F32), // n / o
                        gpu::Elem::Float(gpu::FloatKind::F32), // k / c
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
            AMDArchitecture::Other => vec![],
        }
    }
}
