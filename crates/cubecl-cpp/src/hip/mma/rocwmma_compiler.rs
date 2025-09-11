use crate::{
    hip::{
        HipDialect,
        arch::AMDArchitecture,
        mma::{compile_manual_mma, supported_mma_combinations},
    },
    shared::{
        DialectWmmaCompiler, Flags, Fragment, FragmentIdent, FragmentLayout, ManualMma,
        SupportedMmaCombinations, Variable, WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};
use cubecl_runtime::MmaConfig;

const ROCWMMA_NAMESPACE: &str = "rocwmma";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RocWmmaCompiler {}

impl DialectWmmaCompiler<HipDialect<Self>> for RocWmmaCompiler {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>, _flags: &Flags) -> std::fmt::Result {
        f.write_str("#include <rocwmma/rocwmma.hpp>\n")
    }

    fn compile_wmma_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        flags: &Flags,
    ) -> std::fmt::Result {
        // For manual MMA, maybe add a flag for this at some point
        if flags.elem_bf16 {
            f.write_str("typedef __bf16 bhalf8_t __attribute__((ext_vector_type(8)));\n")?;
            f.write_str("typedef __bf16 bhalf16_t __attribute__((ext_vector_type(16)));\n")?;
        }
        if flags.elem_f16 {
            f.write_str("typedef _Float16 half8_t __attribute__((ext_vector_type(8)));\n")?;
            f.write_str("typedef _Float16 half16_t __attribute__((ext_vector_type(16)));\n")?;
        }
        f.write_str("typedef float float8_t __attribute__((ext_vector_type(8)));\n")
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

    fn compile_manual_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<HipDialect<Self>>,
    ) -> std::fmt::Result {
        compile_manual_mma(f, mma.shape, mma.frag_a, mma.frag_b, mma.frag_c, mma.frag_d)
    }

    fn compile_scaled_mma(
        _f: &mut std::fmt::Formatter<'_>,
        _mma: ManualMma<HipDialect<Self>>,
        _scales_a: Variable<HipDialect<Self>>,
        _scales_b: Variable<HipDialect<Self>>,
        _scales_factor: u32,
    ) -> std::fmt::Result {
        unimplemented!("Scaled MMA not supported in HIP")
    }

    fn supported_wmma_combinations(arch: &AMDArchitecture) -> SupportedMmaCombinations {
        let combinations = match arch {
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 => {
                // For gfx11 the supported tile dimensions are always the same
                //                                   m   n   k
                let tdims = vec![(16, 16, 16), (16, 16, 32)];
                let types = vec![
                    (
                        gpu::ElemType::Float(gpu::FloatKind::F16), // m / i
                        gpu::ElemType::Float(gpu::FloatKind::F32), // n / o
                        gpu::ElemType::Float(gpu::FloatKind::F32), // k / c
                    ),
                    (
                        gpu::ElemType::Float(gpu::FloatKind::F16),
                        gpu::ElemType::Float(gpu::FloatKind::F16),
                        gpu::ElemType::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::ElemType::Float(gpu::FloatKind::F16),
                        gpu::ElemType::Float(gpu::FloatKind::F16),
                        gpu::ElemType::Float(gpu::FloatKind::F16),
                    ),
                    (
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                        gpu::ElemType::Float(gpu::FloatKind::F32),
                        gpu::ElemType::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                        gpu::ElemType::Float(gpu::FloatKind::F32),
                    ),
                    (
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                        gpu::ElemType::Float(gpu::FloatKind::BF16),
                    ),
                ];
                types.into_iter().map(|it| (it, tdims.clone())).collect()
            }
            AMDArchitecture::GFX908 => {
                vec![
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F32), // m / i
                            gpu::ElemType::Float(gpu::FloatKind::F32), // n / o
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                        ), // k / c
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
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                            gpu::ElemType::Float(gpu::FloatKind::F16),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                        ),
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
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                            gpu::ElemType::Float(gpu::FloatKind::F32),
                        ),
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
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                            gpu::ElemType::Float(gpu::FloatKind::BF16),
                        ),
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
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(), // m / i
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(), // n / o
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                        ), // k / c
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
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F16).into(),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                        ),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        (
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                            gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                        ),
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
        };
        combinations
            .into_iter()
            .flat_map(|(ty, sizes)| sizes.into_iter().map(move |size| (ty, size)))
            .map(|((i, o, c), (m, n, k))| MmaConfig {
                a_type: i.into(),
                b_type: o.into(),
                cd_type: c.into(),
                m,
                n,
                k,
            })
            .collect()
    }

    fn supported_mma_combinations(arch: &AMDArchitecture) -> SupportedMmaCombinations {
        supported_mma_combinations(arch)
    }
}
