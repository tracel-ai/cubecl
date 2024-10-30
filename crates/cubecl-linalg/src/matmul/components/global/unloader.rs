use crate::matmul::components::stage::StageWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::GmmConfig;

#[cube]
/// Output to the global matmul
///
/// # Note
///
/// It is only a wrapper over the stage writer because there is no K for the output.
/// Could be deleted in favor of having only the StageWriter
pub trait Unloader<EG: Numeric, G: GmmConfig>: CubeType + 'static + Send + Sync {
    type StageWriter: StageWriter<EG, G>;

    fn as_stage_writer(unloader: Self) -> Self::StageWriter;
}
