use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{
        AttentionPartitionSize, AttentionPrecision, AttentionProblem, AttentionSelection,
        AttentionSetupError, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
        AvailableLineSizes, FlashIdent, args::TensorInputsLaunch, batch::HypercubeSelection,
    },
    kernels::{Algorithm, dummy::DummyAlgorithm},
};

use crate::components::batch::BatchAttentionConfig;
use crate::components::batch::BatchAttentionFamily;
use cubecl_core::frontend::CubePrimitive;

pub enum Strategy {
    /// Temporary implementation
    Tmp,
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, AP: AttentionPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    query: TensorHandle<R, AP::EI>,
    key: TensorHandle<R, AP::EI>,
    value: TensorHandle<R, AP::EI>,
    out: TensorHandle<R, AP::EO>,
) -> Result<(), AttentionSetupError> {
    launch_ref::<R, AP>(
        strategy,
        client,
        &query.as_ref(),
        &key.as_ref(),
        &value.as_ref(),
        &out.as_ref(),
    )
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, AP: AttentionPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::Tmp => launch_tmp::<R, AP>(client, query, key, value, out),
    }
}

pub fn launch_tmp<R: Runtime, AP: AttentionPrecision>(
    client: &ComputeClient<R::Server, R::Channel>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), AttentionSetupError> {
    let line_sizes = AvailableLineSizes::from_elem_types::<R>(
        &AP::EI::as_type_native_unchecked(),
        &AP::EM::as_type_native_unchecked(),
        &AP::EO::as_type_native_unchecked(),
    );
    let line_sizes = DummyAlgorithm::filter_line_sizes(line_sizes)
        .filter_with_tensor(FlashIdent::Query, query.strides, query.shape)
        .filter_with_tensor(FlashIdent::Key, key.strides, key.shape)
        .filter_with_tensor(FlashIdent::Value, value.strides, value.shape)
        .filter_with_tensor(FlashIdent::Out, out.strides, out.shape)
        .pick_max()
        .unwrap();

    let problem = AttentionProblem {
        batch: query.shape[0],
        seq_q: query.shape[1],
        seq_kv: key.shape[1],
        num_heads: query.shape[2],
        head_dim: query.shape[3],
        val_dim: value.shape[3],
        masked: false,
    };

    let tile_size = AttentionTileSize {
        seq_q: 8,
        head_dim: 8,
        seq_kv: 8,
        val_dim: 8,
    };

    let selection = AttentionSelection {
        hypercube_selection: HypercubeSelection {},
        tiling_scheme: AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: 1,
                seq_kv: 1,
                val_dim: 1,
            },
            stage_size: AttentionStageSize { seq_q: 1 },
        },
        plane_dim: 32,
        reuse_key_value: false,
    };

    let config = DummyAlgorithm::setup::<AP, R>(client, &problem, &selection, &line_sizes)?;

    let cube_count_plan = config
        .hypercube_config()
        .cube_count_plan(&problem, &selection);

    unsafe {
        <DummyAlgorithm as Algorithm>::BatchAttention::launch_unchecked::<AP, R>(
            client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                query.as_tensor_arg(line_sizes.query),
                key.as_tensor_arg(line_sizes.key),
                value.as_tensor_arg(line_sizes.value),
            ),
            out.as_tensor_arg(line_sizes.out),
            cube_count_plan.as_args(),
            config,
        );
    }

    Ok(())
}
