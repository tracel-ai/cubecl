use crate::components::tile::{SharedTileConfig, TileConfig};
use crate::components::{MatrixLayout, StageIdent};

/// Execution mode for the RegisterMatmul
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during load
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during load
    Outer,
}

impl ProductType {
    fn from_shared_tile_config(config: SharedTileConfig) -> Self {
        let lhs_preferred = match config.lhs_layout {
            MatrixLayout::RowMajor => ProductType::Inner,
            MatrixLayout::ColMajor => ProductType::Outer,
        };
        let rhs_preferred = match config.lhs_layout {
            MatrixLayout::RowMajor => ProductType::Outer,
            MatrixLayout::ColMajor => ProductType::Inner,
        };

        if lhs_preferred == rhs_preferred {
            lhs_preferred
        } else if config.tile_size.m() == 1 {
            rhs_preferred
        } else if config.tile_size.n() == 1 {
            lhs_preferred
        } else {
            // No better solution
            ProductType::Outer
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterMatmulConfig {
    pub shared: SharedTileConfig,
    pub product_type: ProductType,
}

impl RegisterMatmulConfig {
    pub fn from_shared_tile_config(shared: SharedTileConfig) -> Self {
        Self {
            shared,
            product_type: ProductType::from_shared_tile_config(shared),
        }
    }
}

impl TileConfig for RegisterMatmulConfig {
    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        self.shared.stage_line_size(ident)
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        self.shared.global_line_size(ident)
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        self.shared.matrix_layout(ident)
    }

    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim()
    }
}
