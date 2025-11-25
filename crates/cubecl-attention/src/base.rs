use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{
        AttentionElems, AttentionIdent, AttentionPartitionSize, AttentionProblem,
        AttentionSelection, AttentionSetupError, AttentionStageSize, AttentionTileSize,
        AttentionTilingScheme, AvailableLineSizes,
        args::{TensorArgs, TensorInputsLaunch},
        batch::HypercubeSelection,
    },
    kernels::{Algorithm, blackbox_accelerated::BlackboxAcceleratedAlgorithm, unit::UnitAlgorithm},
};

use crate::components::batch::BatchAttentionConfig;
use crate::components::batch::BatchAttentionFamily;

#[derive(Debug, Clone)]
pub enum Strategy {
    BlackboxAccelerated,
    Unit,
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    query: TensorHandle<R>,
    key: TensorHandle<R>,
    value: TensorHandle<R>,
    mask: Option<TensorHandle<R>>,
    out: TensorHandle<R>,
    attention_elems: AttentionElems,
) -> Result<(), AttentionSetupError> {
    launch_ref(
        strategy,
        client,
        &query.as_ref(),
        &key.as_ref(),
        &value.as_ref(),
        &mask.as_ref().map(|m| m.as_ref()),
        &out.as_ref(),
        &attention_elems,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    attention_elems: &AttentionElems,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::BlackboxAccelerated => launch_attention::<R, BlackboxAcceleratedAlgorithm>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_elems,
        ),
        Strategy::Unit => launch_attention::<R, UnitAlgorithm>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_elems,
        ),
    }
}

pub fn launch_attention<R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    attention_elems: &AttentionElems,
) -> Result<(), AttentionSetupError> {
    let line_sizes = AvailableLineSizes::from_elem_types(
        client,
        query.elem_size,
        attention_elems.mask.size(),
        out.elem_size,
    );
    let line_sizes = A::filter_line_sizes(line_sizes)
        .filter_with_tensor(AttentionIdent::Query, query.strides, query.shape)
        .filter_with_tensor(AttentionIdent::Key, key.strides, key.shape)
        .filter_with_tensor(AttentionIdent::Value, value.strides, value.shape)
        .filter_with_tensor(AttentionIdent::Out, out.strides, out.shape)
        .pick_max()
        .unwrap();

    let problem = AttentionProblem {
        batch: query.shape[0],
        num_heads: query.shape[1],
        seq_q: query.shape[2],
        head_dim: query.shape[3],
        seq_kv: key.shape[2],
        val_dim: value.shape[3],
        masked: mask.is_some(),
        causal: false,
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

    let config = BlackboxAcceleratedAlgorithm::setup(
        client,
        &problem,
        &selection,
        &line_sizes,
        attention_elems,
    )?;

    let cube_count_plan = config
        .hypercube_config()
        .cube_count_plan(&problem, &selection);

    let result = unsafe {
        <BlackboxAcceleratedAlgorithm as Algorithm>::BatchAttention::launch_unchecked::<TensorArgs, R>(
            client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                query.as_tensor_arg(line_sizes.query),
                key.as_tensor_arg(line_sizes.key),
                value.as_tensor_arg(line_sizes.value),
                mask.as_ref()
                    .map(|it| it.as_tensor_arg(line_sizes.out))
                    .into(),
            ),
            out.as_tensor_arg(line_sizes.out),
            cube_count_plan.as_args(),
            config,
            attention_elems,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(AttentionSetupError::Execution(err)),
    }
}
