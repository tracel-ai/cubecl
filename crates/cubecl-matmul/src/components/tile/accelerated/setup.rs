use crate::components::resource::ComputeResources;
use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::tile::{TileConfig, TileMatmulFamily, TileSetupInput};
use crate::components::{
    InvalidConfigError, MatmulChecker, MatmulLineSizes, MatmulPrecision, MatmulProblem, TileSize,
};
use crate::kernels::MatmulAvailabilityError;
use cubecl_core::Feature;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;

impl TileMatmulFamily for AcceleratedMatmul {
    type Matmul<MP: MatmulPrecision> = AcceleratedMatmul;

    fn requires_tensor_cores() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup(
        input: TileSetupInput,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
    ) -> Self::Config {
        let (lhs_line_size, rhs_line_size, stage_line_size_update) =
            if input.vectorization.stage_line_size == 0 {
                (line_sizes.lhs as u32, line_sizes.rhs as u32, false)
            } else {
                (
                    input.vectorization.stage_line_size as u32,
                    input.vectorization.stage_line_size as u32,
                    true,
                )
            };
        AcceleratedConfig::new(
            input.tile_size,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_line_size,
            rhs_line_size,
            line_sizes.out as u32,
        )
    }
}

impl MatmulChecker for AcceleratedMatmul {
    type Config = AcceleratedConfig;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        if config.plane_dim() != 32 {
            return Err(Box::new(
                "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
            ));
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

        let size = config.tile_size();
        if !client.properties().feature_enabled(Feature::Cmma {
            a: es,
            b: es,
            c: ea,
            m: size.m() as u8,
            k: size.k() as u8,
            n: size.n() as u8,
        }) {
            return Err(MatmulAvailabilityError::CmmaInstructionUnavailable {
                input: es,
                output: ea,
                size: Some(TileSize::new(size.m(), size.n(), size.k())),
            });
        }

        if !(MP::ES::is_supported(client) && MP::EA::is_supported(client)) {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: es,
                output: ea,
            });
        }

        Ok(())
    }
}
