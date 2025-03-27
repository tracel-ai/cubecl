use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageEvent {
    /// Before any step
    Begin,
    /// After loading LHS
    LhsLoaded,
    /// After X RHS loads are completed
    RhsLoaded(u32),
    /// After X tile matmul operations are completed
    TmmCompleted(u32),
    /// After the last step
    Finish,
}

#[cube]
pub trait LazyTask: CubeType {
    fn on_event(this: &mut Self, #[comptime] event: StageEvent);
}

#[derive(CubeType)]
pub struct NoTask {}

#[cube]
impl LazyTask for NoTask {
    fn on_event(_this: &mut Self, #[comptime] _event: StageEvent) {
        // Nothing to do
    }
}

impl Default for NoTask {
    fn default() -> Self {
        Self::new()
    }
}

#[cube]
impl NoTask {
    pub fn new() -> NoTask {
        NoTask {}
    }
}
