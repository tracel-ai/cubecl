use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::global::GlobalAttentionConfig;

#[cube]
pub trait AttentionReader<E: Numeric, G: GlobalAttentionConfig> {
    type Stage: CubeType;

    fn init_stage(&mut self, #[comptime] config: G) -> Self::Stage;

    fn read_global(&mut self, stage: &mut Self::Stage, #[comptime] config: G);

    fn advance_view(&mut self);
}
