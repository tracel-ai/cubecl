use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use cubecl_std::tensor::layout::Coords2d;

use crate::components::stage::StageAttentionConfig;
use crate::components::tile::Reducer;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait AttentionPartitioner: Send + Sync + 'static {
    type Reducer: Reducer;

    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates<S: StageAttentionConfig>(#[comptime] config: S) -> Coords2d;
}
