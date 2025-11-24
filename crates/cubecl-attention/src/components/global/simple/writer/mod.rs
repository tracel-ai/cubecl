use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use cubecl_matmul::components::global::{GlobalWriterConfig, PartitionedStage, WriteEventListener};

mod plane;
mod unit;

use cubecl_std::tensor::{View, layout::Coords2d};
pub use plane::*;
pub use unit::*;

use crate::components::stage::StageAttentionConfig;

#[cube]
pub trait AttentionWriter<ES: Numeric, EG: Numeric>: WriteEventListener {
    fn init<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    fn stage(&mut self) -> PartitionedStage<ES>;
}
