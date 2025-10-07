use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{
        AttentionIdent, AttentionPartitionSize, AttentionPrecision, AttentionProblem,
        AttentionSelection, AttentionSetupError, AttentionStageSize, AttentionTileSize,
        AttentionTilingScheme, AvailableLineSizes, args::TensorInputsLaunch, attention_types::*,
        batch::HypercubeSelection,
    },
    kernels::{Algorithm, dummy::DummyRegisterAlgorithm},
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
    query: TensorHandle<R, QG<AP>>,
    key: TensorHandle<R, KG<AP>>,
    value: TensorHandle<R, VG<AP>>,
    out: TensorHandle<R, OG<AP>>,
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
        &QG::<AP>::as_type_native_unchecked(),
        &MSK::<AP>::as_type_native_unchecked(),
        &OG::<AP>::as_type_native_unchecked(),
    );
    let line_sizes = DummyRegisterAlgorithm::filter_line_sizes(line_sizes)
        .filter_with_tensor(AttentionIdent::Query, query.strides, query.shape)
        .filter_with_tensor(AttentionIdent::Key, key.strides, key.shape)
        .filter_with_tensor(AttentionIdent::Value, value.strides, value.shape)
        .filter_with_tensor(AttentionIdent::Out, out.strides, out.shape)
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
        two_rows_in_array_tile: false,
    };

    let config = DummyRegisterAlgorithm::setup::<AP, R>(client, &problem, &selection, &line_sizes)?;

    let cube_count_plan = config
        .hypercube_config()
        .cube_count_plan(&problem, &selection);

    unsafe {
        <DummyRegisterAlgorithm as Algorithm>::BatchAttention::launch_unchecked::<AP, R>(
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
