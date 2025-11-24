use cubecl_core::ir::MatrixIdent;

use crate::components::tile::{SharedTileConfig, TileConfig};

use crate::components::StageIdent;
use crate::components::stage::SwizzleMode;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadMethod {
    Manual,
    LoadMatrix,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StoreMethod {
    Manual,
    StoreMatrix,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaMatmulConfig {
    pub shared: SharedTileConfig,
    lhs_load_method: LoadMethod,
    rhs_load_method: LoadMethod,
    acc_load_method: LoadMethod,
    store_method: StoreMethod,
}

impl MmaMatmulConfig {
    pub fn from_shared_tile_config(
        shared: SharedTileConfig,
        lhs_load_method: LoadMethod,
        rhs_load_method: LoadMethod,
        acc_load_method: LoadMethod,
        store_method: StoreMethod,
    ) -> Self {
        Self {
            shared,
            lhs_load_method,
            rhs_load_method,
            acc_load_method,
            store_method,
        }
    }

    pub fn load_method(&self, ident: MatrixIdent) -> LoadMethod {
        match ident {
            MatrixIdent::A => self.lhs_load_method,
            MatrixIdent::B => self.rhs_load_method,
            MatrixIdent::Accumulator => self.acc_load_method,
        }
    }

    pub fn store_method(&self) -> StoreMethod {
        self.store_method
    }
}

impl TileConfig for MmaMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim()
    }

    fn elements_in_tile_m(&self) -> u32 {
        self.shared.elements_in_tile_m()
    }

    fn elements_in_tile_n(&self) -> u32 {
        self.shared.elements_in_tile_n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.shared.elements_in_tile_k()
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        self.shared.swizzle_mode(ident)
    }
}
