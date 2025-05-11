use cubecl_core as cubecl;
use cubecl_core::prelude::CubeType;

#[derive(Copy, Clone, CubeType)]
pub enum BufferId {
    A,
    B,
}

impl BufferId {
    pub fn to_index(&self) -> u32 {
        match self {
            BufferId::A => 0,
            BufferId::B => 1,
        }
    }
}

#[derive(CubeType, Clone)]
pub struct TaskCounter {
    #[cube(comptime)]
    pub counter: u32,
}
