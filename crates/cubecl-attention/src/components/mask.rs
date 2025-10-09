use crate::components::global::dummy::MaskReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::{Coordinates, Coords2d};
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::global::GlobalAttentionConfig;
use crate::components::{AttentionPrecision, AttentionTilingScheme};

#[derive(CubeType, Copy, Clone)]
pub struct LogicalMask {
    #[cube(comptime)]
    pub causal: bool,
    pub out_of_bounds: CubeOption<Coords2d>,
}

#[cube]
impl LogicalMask {
    pub fn apply<E: Numeric>(&self, pos: Coords2d) -> E {
        let causal_masked = self.causal && pos.0 < pos.1;

        let oob_masked = match self.out_of_bounds {
            CubeOption::Some(bounds) => !Coords2d::is_in_bounds(&pos, &bounds),
            CubeOption::None => false,
        };

        E::cast_from(causal_masked || oob_masked) * E::min_value()
    }
}

#[derive(CubeType)]
pub enum Mask<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    /// Full mask tensor in global memory.
    /// Used when the user provides an explicit mask.
    /// Causal or out-of-bounds padding are applied directly in the materialized mask
    Materialized(MaskReader<AP, G>, LogicalMask),

    /// Mask is applied logically.
    /// This variant is chosen when no mask tensor is provided but the attention logic
    /// requires masking for causal or padding purposes.
    Logical(LogicalMask),

    /// No mask is applied at all.
    /// Used when neither a mask tensor is provided nor causal/padding masking is needed.
    None,
}

#[derive(CubeType, Copy, Clone)]
pub struct GlobalMask {
    origin: Coords2d,
    logical_mask: LogicalMask,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct StageMask {
    origin: Coords2d,
    logical_mask: LogicalMask,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct PartitionMask {
    origin: Coords2d,
    logical_mask: LogicalMask,
    #[cube(comptime)]
    tiling_scheme: AttentionTilingScheme,
}

#[derive(CubeType, Copy, Clone)]
pub struct TileMask {
    origin: Coords2d,
    logical_mask: LogicalMask,
}

#[cube]
impl GlobalMask {
    pub fn new(
        logical_mask: LogicalMask,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> GlobalMask {
        GlobalMask {
            origin: (0u32, 0u32).runtime(),
            logical_mask,
            tiling_scheme,
        }
    }

    pub fn to_stage(&self, row: u32, col: u32) -> StageMask {
        let q_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_q());
        let kv_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_kv());

        StageMask {
            origin: Coords2d::add(self.origin, (row * q_factor, col * kv_factor)),
            logical_mask: self.logical_mask,
            tiling_scheme: self.tiling_scheme,
        }
    }
}

#[cube]
impl StageMask {
    pub fn to_partition(&self, row: u32) -> PartitionMask {
        let q_factor = comptime!(self.tiling_scheme.elements_in_partition_seq_q());

        PartitionMask {
            origin: Coords2d::add(self.origin, (row * q_factor, 0u32)),
            logical_mask: self.logical_mask,
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
            origin: Coords2d::add(self.origin, (row * q_factor, col * kv_factor)),
            logical_mask: self.logical_mask,
        }
    }
}

#[cube]
impl TileMask {
    pub fn apply<E: Numeric>(&self, pos: Coords2d) -> E {
        self.logical_mask
            .apply::<E>(Coords2d::add(self.origin, pos))
    }
}
