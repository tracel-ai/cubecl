use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use cubecl_matmul::components::global::{
    PartitionedStage, WriteEventListener, memory::GlobalMemoryConfig,
};

mod plane;
mod unit;

use cubecl_std::tensor::{View, layout::Coords2d};
pub use plane::*;
pub use unit::*;

use crate::components::stage::StageAttentionConfig;

#[cube]
pub trait AttentionWriter<ES: Numeric, EG: Numeric>: WriteEventListener {
    fn new<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] global_config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self;

    fn stage(&mut self) -> PartitionedStage<ES>;
}
