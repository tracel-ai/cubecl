use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::stage::StageConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Events that occur during the process of loading tiles to
/// registers and executing inner Tile Matmuls
pub enum StageEvent {
    /// Before any step
    Begin,
    /// After loading LHS
    LhsLoaded { current: u32, total: u32 },
    /// After X RHS loads are completed
    RhsLoaded { current: u32, total: u32 },
    /// After X tile matmul operations are completed
    TileMatmulCompleted { current: u32, total: u32 },
    /// After the last step
    Finish,
}

#[cube]
/// Function that is called at each [StageEvent]
pub trait StageEventListener<S: StageConfig>: CubeType {
    fn on_event(this: &mut Self, #[comptime] event: StageEvent, #[comptime] config: S);
}

#[derive(CubeType)]
/// Use when there is no event listening to do
pub struct NoEvent {}

#[cube]
impl<S: StageConfig> StageEventListener<S> for NoEvent {
    fn on_event(_this: &mut Self, #[comptime] _event: StageEvent, #[comptime] _config: S) {
        // Nothing to do
    }
}

impl Default for NoEvent {
    fn default() -> Self {
        Self::new()
    }
}

#[cube]
impl NoEvent {
    pub fn new() -> NoEvent {
        NoEvent {}
    }
}
