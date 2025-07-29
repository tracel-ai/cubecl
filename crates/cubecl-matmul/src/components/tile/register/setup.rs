use std::fmt::Display;

use crate::components::error::MatmulSetupError;
use crate::components::resource::ComputeResources;
use crate::components::tile::register::config::RegisterConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::tile::{TileMatmulFamily, TileSetupInfo};
use crate::components::{AvailableLineSizes, InvalidConfigError, MatmulPrecision};
use cubecl_core::prelude::*;

impl TileMatmulFamily for RegisterMatmul {
    type Matmul<MP: MatmulPrecision> = RegisterMatmul;
    type Config = RegisterConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_setup_info: TileSetupInfo,
    ) -> Result<Self::Config, MatmulSetupError> {
        RegisterConfig::new::<MP, R>(
            client,
            tile_setup_info.tile_size,
            tile_setup_info.plane_dim,
            tile_setup_info.lhs_layout,
            tile_setup_info.rhs_layout,
            tile_setup_info.lhs_line_size as u32,
            tile_setup_info.rhs_line_size as u32,
            tile_setup_info.out_line_size as u32,
            tile_setup_info.lhs_line_size as u32,
            tile_setup_info.rhs_line_size as u32,
        )
    }

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
            .filter_lhs(|ls| *ls <= 4)
            .filter_rhs(|ls| *ls <= 4)
            .filter_out(|ls| *ls <= 4)
    }
}

pub struct RegisterMatmulConfigError {
    func: Box<dyn Fn() -> String>,
}

impl Display for RegisterMatmulConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}
