use std::fmt::Display;

use crate::components::resource::ComputeResources;
use crate::components::tile::register::config::RegisterConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::tile::{TileConfig, TileMatmulFamily};
use crate::components::{
    AvailableLineSizes, Ident, InvalidConfigError, MatmulChecker, MatmulPrecision, MatmulProblem,
    MatrixLayout,
};
use crate::kernels::matmul::{MatmulSelection, unit_matmul_selection};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_core::Feature;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;

impl TileMatmulFamily for RegisterMatmul {
    type Matmul<MP: MatmulPrecision> = RegisterMatmul;

    fn requires_tensor_cores() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let max_lhs = match problem.lhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_k(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_m(),
        } as u8;
        let max_rhs = match problem.rhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_n(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_k(),
        } as u8;
        let max_out = selection.tiling_scheme.elements_in_tile_n() as u8;

        let lhs_global_line_size = available_line_sizes.maximize_lhs(problem, Some(max_lhs));
        let rhs_global_line_size = available_line_sizes.maximize_rhs(problem, Some(max_rhs));
        let out_global_line_size = available_line_sizes.maximize_out(problem, Some(max_out));

        let stage_vectorization = selection.stage_vectorization;

        let (lhs_stage_line_size, rhs_stage_line_size, stage_line_size_update) =
            if stage_vectorization.stage_line_size == 0 {
                (
                    lhs_global_line_size as u32,
                    rhs_global_line_size as u32,
                    false,
                )
            } else {
                (
                    stage_vectorization.stage_line_size as u32,
                    stage_vectorization.stage_line_size as u32,
                    true,
                )
            };

        Ok(RegisterConfig::new(
            selection.tiling_scheme,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_global_line_size as u32,
            rhs_global_line_size as u32,
            out_global_line_size as u32,
            lhs_stage_line_size as u32,
            rhs_stage_line_size as u32,
        ))
    }

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
    ) -> MatmulSelection {
        unit_matmul_selection(problem, plane_dim)
    }
}

pub struct RegisterMatmulConfigError {
    func: Box<dyn Fn() -> String>,
}

impl RegisterMatmulConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + 'static>(func: F) -> Box<dyn Display> {
        Box::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for RegisterMatmulConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}
impl MatmulChecker for RegisterMatmul {
    type Config = RegisterConfig;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let m = config.tile_size().m();
        let n = config.tile_size().n();
        let k = config.tile_size().k();

        // 128 a bit arbitrary, but accepts 4x4x4 and rejects 8x8x8
        if m * n * k > 128 {
            return Err(RegisterMatmulConfigError::new(move || {
                format!(
                    "Tile size m-n-k={:?}-{:?}-{:?} is too large for register matmul",
                    m, n, k
                )
            }));
        }

        let lhs = config.stage_line_size(Ident::Lhs);
        let rhs = config.stage_line_size(Ident::Rhs);

        match config.matrix_layout(Ident::Lhs) {
            MatrixLayout::RowMajor => {
                if k % lhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            k, lhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if m % lhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            m, lhs
                        )
                    }));
                }
            }
        }
        match config.matrix_layout(Ident::Rhs) {
            MatrixLayout::RowMajor => {
                if n % rhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            n, rhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if k % rhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            k, rhs
                        )
                    }));
                }
            }
        }

        Ok(())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        if config.stage_dynamic_line_size
            && !client
                .properties()
                .feature_enabled(Feature::DynamicLineSize)
        {
            return Err(MatmulAvailabilityError::DynamicLineSizeUnavailable);
        }

        let es = MP::ES::as_elem_native().expect("to be a native type");
        let ea = MP::EA::as_elem_native().expect("to be a native type");

        let es = match es {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => es,
        };

        let ea = match ea {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => ea,
        };

        if !(MP::ES::is_supported(client) && MP::EA::is_supported(client)) {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: es,
                output: ea,
            });
        }

        Ok(())
    }
}
