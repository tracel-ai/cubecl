use crate::matmul::matmul_stage::StageWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::GmmConfig;

#[cube]
pub trait Unloader<EG: Numeric, G: GmmConfig>: CubeType + 'static + Send + Sync {
    type StageWriter: StageWriter<EG, G>;

    fn as_stage_writer(unloader: Self) -> Self::StageWriter;
}
