use crate::matmul::components::global::load::{
    AsyncBufferLoadingStrategy, AsyncFullLoadingStrategy, SyncBufferLoadingStrategy,
    SyncFullLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, Quantization};
use crate::matmul::components::stage::Stage;
use crate::matmul::components::{InputIdent, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
pub trait LoadingInfo: CubeType + Copy + Clone {
    fn num_tasks(this: &Self) -> u32;
}

#[cube]
pub(crate) fn default_sync_full_load<
    LS: SyncFullLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: &TensorReader<MP::EI>,
    stage: &mut Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let loading_info = LS::preliminary_computation::<G>(input_ident, config);

    for task_id in 0..LS::LoadingInfo::num_tasks(&loading_info) {
        LS::load_task::<MP, G>(
            task_id,
            loading_info,
            read_view,
            stage,
            quantization,
            input_ident,
            config,
        );
    }
}

#[cube]
pub(crate) fn default_sync_buffer_load<
    LS: SyncBufferLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: &TensorReader<MP::EI>,
    stage: &mut Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    todo!()
}

#[cube]
pub(crate) fn default_async_full_load<
    LS: AsyncFullLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: &TensorReader<MP::EI>,
    stage: &mut Stage<MP::ES, LS::TilingLayout>,
    mechanism: &CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    todo!()
}

#[cube]
pub(crate) fn default_async_buffer_load<
    LS: AsyncBufferLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: &TensorReader<MP::EI>,
    stage: &mut Stage<MP::ES, LS::TilingLayout>,
    mechanism: &CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    todo!()
}
