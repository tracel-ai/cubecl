use crate::components::error::MatmulSetupError;
use crate::components::resource::ComputeResources;
use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::tile::{TileMatmulFamily, TileSetupInfo};
use crate::components::{InvalidConfigError, MatmulPrecision};
use cubecl_core::prelude::*;

impl TileMatmulFamily for AcceleratedMatmul {
    type Matmul<MP: MatmulPrecision> = AcceleratedMatmul;
    type Config = AcceleratedConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_setup_info: TileSetupInfo,
    ) -> Result<Self::Config, MatmulSetupError> {
        AcceleratedConfig::new::<MP, R>(
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
}
