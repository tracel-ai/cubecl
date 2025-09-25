use crate::components::{InputPrecision, global::WriteEventListener, stage::Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<IP: InputPrecision>:
    WriteEventListener + CubeType + 'static + Send + Sync
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage, ReadWrite>;

    fn stage(this: &Self) -> Self::Stage;
}
