use crate::matmul::matmul_global::WriteView;
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Unloader<EG: Numeric>: CubeType + 'static + Send + Sync {
    type WriteView: WriteView<EG>;
    type StageWriter: StageWriter<EG>;

    fn new(
        gmem: <Self::WriteView as WriteView<EG>>::Global,
        #[comptime] stage_info: StageInfo,
    ) -> Self;

    fn unload(unloader: Self) -> Self::StageWriter;
}
