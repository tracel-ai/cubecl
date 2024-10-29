use std::str::FromStr;

use cubecl_core::{
    ir::{Elem, FloatKind},
    Feature,
};
use cubecl_runtime::DeviceProperties;

pub enum AMDArchitecture {
    // RDNA
    // gfx1100, gfx1101, gfx1102
    GFX11,
    // CDNA
    GFX908,
    GFX90A,
    // gfx940, gfx941, gfx942
    GFX94,
    // Not particularly specific architecture
    Other,
}

impl FromStr for AMDArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm = s.to_lowercase();
        if norm.starts_with("gfx11") {
            Ok(AMDArchitecture::GFX11)
        } else if norm == "gfx908" {
            Ok(AMDArchitecture::GFX908)
        } else if norm == "gfx90a" {
            Ok(AMDArchitecture::GFX90A)
        } else if norm.starts_with("gfx94") {
            Ok(AMDArchitecture::GFX94)
        } else {
            Ok(AMDArchitecture::Other)
        }
    }
}

impl AMDArchitecture {
    pub fn warp_size(&self) -> i32 {
        // CDNA supports wave64 (gfx9 and gfx940+) and RDNA wave32 (gfx11)
        match self {
            AMDArchitecture::GFX11 => 32,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94 => 64,
            AMDArchitecture::Other => 0,
        }
    }

    // TODO verify that rocWMMA is installed on the system
    pub fn is_wmma_capable(&self) -> bool {
        match self {
            AMDArchitecture::GFX11
            | AMDArchitecture::GFX908
            | AMDArchitecture::GFX90A
            | AMDArchitecture::GFX94 => true,
            AMDArchitecture::Other => false,
        }
    }

    // Reference: https://github.com/ROCm/rocWMMA/blob/develop/docs/api-reference/api-reference-guide.rst
    //
    //                                        i     o    c    m   n   k
    fn get_wmma_combinations(&self) -> Vec<(Elem, Elem, Elem, Vec<(u8, u8, u8)>)> {
        match self {
            AMDArchitecture::GFX11 => {
                // For gfx11 the supported tile dimensions are always the same
                //                                   m   n   k
                let tdims = vec![(16, 16, 16), (16, 16, 32)];
                let types = vec![
                    (
                        Elem::Float(FloatKind::F16), // i
                        Elem::Float(FloatKind::F32), // o
                        Elem::Float(FloatKind::F32), // c
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F32),
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
                        Elem::Float(FloatKind::F32),
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
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
                        Elem::Float(FloatKind::F32), // i
                        Elem::Float(FloatKind::F32), // o
                        Elem::Float(FloatKind::F32), // c
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
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F32),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
                        Elem::Float(FloatKind::F32),
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
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
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
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
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
                        Elem::Float(FloatKind::F32), // i
                        Elem::Float(FloatKind::F32), // o
                        Elem::Float(FloatKind::F32), // c
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
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F32),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        Elem::Float(FloatKind::F16),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::F32),
                        vec![
                            (16, 16, 16),
                            (16, 16, 32),
                            (32, 32, 8),
                            (32, 32, 16),
                            (32, 32, 32),
                        ],
                    ),
                    (
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
                        Elem::Float(FloatKind::BF16),
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

    pub fn register_wmma_features(&self, properties: &mut DeviceProperties<Feature>) {
        if !self.is_wmma_capable() {
            return;
        }
        properties.register_feature(Feature::CmmaWarpSize(self.warp_size()));
        for (i, o, c, tdims) in self.get_wmma_combinations() {
            for (m, n, k) in tdims {
                properties.register_feature(Feature::Cmma {
                    a: i,
                    b: o,
                    c,
                    m,
                    n,
                    k,
                });
            }
        }
    }
}
