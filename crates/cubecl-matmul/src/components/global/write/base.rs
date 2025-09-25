use crate::components::{
    InputPrecision,
    global::{WriteEventListener, WriteTiling, memory::GlobalMemoryConfig},
    stage::{Stage, StageConfig, StageFamily},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

pub trait GlobalWriterFamily: 'static + Send + Sync {
    type Stage: StageFamily<ReadWrite>;
    type Writer<IP: InputPrecision>: GlobalWriter<
            IP,
            Stage = <Self::Stage as StageFamily<ReadWrite>>::Stage<IP::Stage, WriteTiling>,
        >;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<IP: InputPrecision>:
    WriteEventListener + CubeType + 'static + Send + Sync
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage, ReadWrite>;

    /// Init this writer from a global tensor and config
    fn init<S: StageConfig>(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self;

    /// Stage used by this writer
    fn stage(this: &Self) -> Self::Stage;
}
