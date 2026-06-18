use crate::{hip::arch::AMDArchitecture, shared::SupportedMmaCombinations};
use cubecl_core::ir::{ElemType, FloatKind, IntKind, features::MmaConfig};

pub(super) fn compile_rocwmma_includes() -> String {
    "#include <rocwmma/rocwmma.hpp>\n".into()
}

pub(super) fn supported_wmma_combinations_rocwmma(
    arch: &AMDArchitecture,
) -> SupportedMmaCombinations {
    let combinations = match arch {
        AMDArchitecture::GFX12 => {
            // Group types by their tile dimensions for readability
            let tdims_16_16_32 = vec![(16, 16, 32)];
            let types_16_16_32 = vec![
                (
                    ElemType::Float(FloatKind::E5M2), // bfloat8_t / bf8
                    ElemType::Float(FloatKind::F32),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::E4M3), // float8_t / f8
                    ElemType::Float(FloatKind::F32),
                    ElemType::Float(FloatKind::F32),
                ),
            ];

            let tdims_16_16_16 = vec![(16, 16, 16)];
            let types_16_16_16 = vec![
                (
                    ElemType::Int(IntKind::I8),
                    ElemType::Int(IntKind::I32),
                    ElemType::Int(IntKind::I32),
                ),
                (
                    ElemType::Int(IntKind::I8),
                    ElemType::Int(IntKind::I8),
                    ElemType::Int(IntKind::I32),
                ),
                (
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F32),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::F32),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                ),
            ];

            // Combine all type-dimension pairs
            types_16_16_32
                .into_iter()
                .map(|it| (it, tdims_16_16_32.clone()))
                .chain(
                    types_16_16_16
                        .into_iter()
                        .map(|it| (it, tdims_16_16_16.clone())),
                )
                .collect()
        }
        AMDArchitecture::GFX10 | AMDArchitecture::GFX11 => {
            // For gfx11 the supported tile dimensions are always the same
            //                                   m   n   k
            let tdims = vec![(16, 16, 16), (16, 16, 32)];
            let types = vec![
                (
                    ElemType::Float(FloatKind::F16), // m / i
                    ElemType::Float(FloatKind::F32), // n / o
                    ElemType::Float(FloatKind::F32), // k / c
                ),
                (
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                    ElemType::Float(FloatKind::F16),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::F32),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::F32),
                ),
                (
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                    ElemType::Float(FloatKind::BF16),
                ),
            ];
            types.into_iter().map(|it| (it, tdims.clone())).collect()
        }
        AMDArchitecture::GFX908 => {
            vec![
                (
                    (
                        ElemType::Float(FloatKind::F32), // m / i
                        ElemType::Float(FloatKind::F32), // n / o
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F32),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::F32),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
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
                        ElemType::Float(FloatKind::F32), // m / i
                        ElemType::Float(FloatKind::F32), // n / o
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F32),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
                        ElemType::Float(FloatKind::F16),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::F32),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::F32),
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
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
                        ElemType::Float(FloatKind::BF16),
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
