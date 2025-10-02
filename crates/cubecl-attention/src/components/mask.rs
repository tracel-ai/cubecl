use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionTilingScheme;

#[derive(CubeType, Copy, Clone)]
pub struct GlobalMask {
    q_bound: u32,
    kv_bound: u32,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct StageMask {
    q_bound: u32,
    kv_bound: u32,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct PartitionMask {
    q_bound: u32,
    kv_bound: u32,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct TileMask {
    q_bound: u32,
    kv_bound: u32,
}

#[cube]
impl GlobalMask {
    pub fn new(
        q_bound: u32,
        kv_bound: u32,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> GlobalMask {
        GlobalMask {
            q_bound,
            kv_bound,
            tiling_scheme,
        }
    }

    pub fn to_stage(&self, row: u32, col: u32) -> StageMask {
        let q_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_q());
        let kv_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_kv());

        StageMask {
            q_bound: self.q_bound.saturating_sub(row * q_factor),
            kv_bound: self.kv_bound.saturating_sub(col * kv_factor),
            tiling_scheme: self.tiling_scheme,
        }
    }
}

#[cube]
impl StageMask {
    pub fn to_partition(&self, row: u32) -> PartitionMask {
        let q_factor = comptime!(self.tiling_scheme.elements_in_partition_seq_q());

        PartitionMask {
            q_bound: self.q_bound.saturating_sub(row * q_factor),
            kv_bound: self.kv_bound,
            tiling_scheme: self.tiling_scheme,
        }
    }
}

#[cube]
impl PartitionMask {
    pub fn to_tile(self, row: u32, col: u32) -> TileMask {
        let q_factor = comptime!(self.tiling_scheme.elements_in_tile_seq_q());
        let kv_factor = comptime!(self.tiling_scheme.elements_in_tile_seq_kv());

        TileMask {
            q_bound: self.q_bound.saturating_sub(row * q_factor),
            kv_bound: self.kv_bound.saturating_sub(col * kv_factor),
        }
    }
}

#[cube]
impl TileMask {
    pub fn apply<E: Numeric>(&self, pos: Coords2d) -> E {
        let should_mask = E::cast_from(pos.0 >= self.q_bound || pos.1 >= self.kv_bound);
        should_mask * E::min_value()
    }
}
