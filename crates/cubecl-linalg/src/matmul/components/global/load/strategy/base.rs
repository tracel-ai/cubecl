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
pub trait LoadingJob<MP: MatmulPrecision>: CubeType + Copy + Clone {
    fn execute_task<G: GlobalConfig>(this: &mut Self, task_id: u32, #[comptime] config: G);
    fn len(this: &Self) -> u32;
}

#[cube]
pub(crate) fn default_sync_full_load<
    LS: SyncFullLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, G>(read_view, stage, quantization, input_ident, config);

    for task_id in 0..LS::Job::len(&job) {
        LS::Job::execute_task::<G>(&mut job, task_id, config);
    }
}

#[cube]
pub(crate) fn default_sync_buffer_load<
    LS: SyncBufferLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] buffer_index: u32,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, G>(
        read_view,
        stage,
        quantization,
        buffer_index,
        input_ident,
        config,
    );

    for task_id in 0..LS::Job::len(&job) {
        LS::Job::execute_task::<G>(&mut job, task_id, config);
    }
}

#[cube]
pub(crate) fn default_async_full_load<
    LS: AsyncFullLoadingStrategy<MP, CM>,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    mechanism: CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<G>(
        read_view,
        stage,
        mechanism,
        quantization,
        input_ident,
        config,
    );

    for task_id in 0..LS::Job::len(&job) {
        LS::Job::execute_task::<G>(&mut job, task_id, config);
    }
}

#[cube]
pub(crate) fn default_async_buffer_load<
    LS: AsyncBufferLoadingStrategy<MP, CM>,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    mechanism: CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] buffer_index: u32,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<G>(
        read_view,
        stage,
        mechanism,
        quantization,
        buffer_index,
        input_ident,
        config,
    );

    for task_id in 0..LS::Job::len(&job) {
        LS::Job::execute_task::<G>(&mut job, task_id, config);
    }
}
