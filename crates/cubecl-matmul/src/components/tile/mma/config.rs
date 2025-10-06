use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::Numeric;
use cubecl_runtime::MmaConfig;

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{MatrixLayout, StageIdent, TileSize};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated Matmul
pub struct MmaMatmulConfig {
    tile_size: TileSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_global_line_size: u32,
    rhs_global_line_size: u32,
    out_global_line_size: u32,
    lhs_stage_line_size: u32,
    rhs_stage_line_size: u32,
}

impl TileConfig for MmaMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        match ident {
            StageIdent::Lhs => self.lhs_layout,
            StageIdent::Rhs => self.rhs_layout,
            StageIdent::Acc => MatrixLayout::RowMajor,
            StageIdent::Out => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_stage_line_size,
            StageIdent::Rhs => self.rhs_stage_line_size,
            StageIdent::Acc => self.out_global_line_size,
            StageIdent::Out => self.out_global_line_size,
        }
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_global_line_size,
            StageIdent::Rhs => self.rhs_global_line_size,
            StageIdent::Acc => self.out_global_line_size,
            StageIdent::Out => self.out_global_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.tile_size
    }
}

impl MmaMatmulConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for accelerated matmul
    ///
    /// May return an error if:
    /// - cmma is unavailable
    /// - cmma is unavailable for given types
    pub fn new<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_size: TileSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_global_line_size: u32,
        rhs_global_line_size: u32,
        out_global_line_size: u32,
        lhs_stage_line_size: u32,
        rhs_stage_line_size: u32,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            tile_size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_global_line_size,
            rhs_global_line_size,
            out_global_line_size,
            lhs_stage_line_size,
            rhs_stage_line_size,
        }
        .check_availability::<Lhs, Rhs, Acc, R>(client)
    }

    fn check_availability<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        let lhs = Lhs::as_type_native_unchecked();
        let rhs = Rhs::as_type_native_unchecked();
        let acc = Acc::as_type_native_unchecked();

        let size = self.tile_size();
        if !client.properties().features.mma.contains(&MmaConfig {
            a_type: lhs,
            b_type: rhs,
            cd_type: acc,
            m: size.m(),
            k: size.k(),
            n: size.n(),
        }) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        Ok(self)
    }
}
