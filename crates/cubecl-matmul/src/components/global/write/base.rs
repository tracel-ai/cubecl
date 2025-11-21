use crate::components::{
    MatrixPrecision,
    global::{RoleRuleConfig, WriteEventListener, WriteTiling, memory::GlobalMemoryConfig},
    stage::{Stage, StageFamily, StageMemoryConfig},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

pub trait GlobalWriterFamily: 'static + Send + Sync {
    type Stage: StageFamily<ReadWrite>;
    type Writer<IP: MatrixPrecision>: GlobalWriter<
            IP,
            Stage = <Self::Stage as StageFamily<ReadWrite>>::Stage<IP::Stage, WriteTiling>,
        >;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<IP: MatrixPrecision>:
    WriteEventListener + CubeType + 'static + Send + Sync
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage, ReadWrite>;

    /// Init this writer from a global tensor and config
    fn init(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    /// Stage used by this writer
    fn stage(this: &Self) -> Self::Stage;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalWriterConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub role_rule_config: RoleRuleConfig,
    pub plane_dim: u32,
}
