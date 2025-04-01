use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::{
    CopyMechanism, GlobalConfig, multi_stage::double_buffering::BufferId,
};

#[cube]
pub trait BufferLoader<EG: Numeric, ES: Numeric, G: GlobalConfig>:
    CubeType + 'static + Send + Sync
{
    type StageReader: CubeType;

    // TODO maybe buffer cannot be comptime in the future
    fn reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader;

    fn advance_view(this: &mut Self, k_offset: u32);
}

#[cube]
pub trait SyncBufferLoader<EG: Numeric, ES: Numeric, G: GlobalConfig>:
    BufferLoader<EG, ES, G>
{
    /// Fills the buffer at the current k offset.
    fn fill_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: G);
}

#[cube]
pub trait AsyncBufferLoader<EG: Numeric, ES: Numeric, G: GlobalConfig>:
    BufferLoader<EG, ES, G>
{
    /// Fills the buffer at the current k offset.
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: G,
    );

    /// Fills the specified buffer with zeros
    fn clear_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: G);
}
