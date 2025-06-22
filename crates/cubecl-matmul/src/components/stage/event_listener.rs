use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::stage::StageConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageEvent {
    /// Before any step
    Begin,
    /// After loading LHS
    LhsLoaded { current: u32, total: u32 },
    /// After X RHS loads are completed
    RhsLoaded { current: u32, total: u32 },
    /// After X tile matmul operations are completed
    TmmCompleted { current: u32, total: u32 },
    /// After the last step
    Finish,
}

#[cube]
pub trait StageEventListener<S: StageConfig>: CubeType {
    fn on_event(this: &mut Self, #[comptime] event: StageEvent, #[comptime] config: S);
}

#[derive(CubeType)]
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
