use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Unloader<E: Numeric>: CubeType + 'static + Send + Sync {
    type GlobalView: GlobalView<E>;
    type StageWriter: StageWriter<E>;

    fn new(
        gmem: <Self::GlobalView as GlobalView<E>>::Global,
        #[comptime] stage_info: StageInfo,
    ) -> Self;

    fn unload(unloader: Self) -> Self::StageWriter;
}
