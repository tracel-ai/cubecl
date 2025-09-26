use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType, Debug, Clone, Copy, PartialEq, Eq)]
/// Events that occur during the process of storing tiles to
/// a stage and executing writes
pub enum WriteEvent {
    /// Before any step
    Begin,
    /// After each tile is stored into the stage
    TileStored { tile: Coords2d },
    /// After the last step
    Finish,
}

#[cube]
/// Function that is called at each [WriteEvent]
pub trait WriteEventListener: CubeType {
    fn on_event(this: &mut Self, event: WriteEvent);
}
