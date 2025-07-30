use cubecl_core as cubecl;
use cubecl_core::prelude::CubeType;

#[derive(Copy, Clone, CubeType)]
/// Identifier for the stage in global double buffering
pub enum StageBuffer {
    /// First buffer
    A,
    /// Second buffer
    B,
}

impl StageBuffer {
    pub fn to_index(&self) -> u32 {
        match self {
            StageBuffer::A => 0,
            StageBuffer::B => 1,
        }
    }
}

#[derive(CubeType, Clone)]
/// Comptime counter for loading tasks
pub struct TaskCounter {
    #[cube(comptime)]
    pub counter: u32,
}
