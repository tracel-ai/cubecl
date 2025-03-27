use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait LazyTask: CubeType {
    fn execute(this: &mut Self, #[comptime] task_id: u32);
}

#[derive(CubeType)]
pub struct NoTask {}

#[cube]
impl LazyTask for NoTask {
    fn execute(_this: &mut Self, #[comptime] _task_id: u32) {}
}

#[cube]
impl NoTask {
    pub fn new() -> NoTask {
        NoTask {}
    }
}
