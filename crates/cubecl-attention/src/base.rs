use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{
        AttentionIdent, AttentionProblem, AttentionSetupError, AttentionStorageTypes,
        AvailableLineSizes,
        args::{TensorArgs, TensorInputsLaunch},
    },
    kernels::{
        Algorithm, SharedAttentionSettings, blackbox_accelerated::BlackboxAcceleratedAlgorithm,
        unit::UnitAlgorithm,
    },
};

use crate::components::batch::BatchAttentionFamily;

#[derive(Debug, Clone)]
pub enum Strategy {
    BlackboxAccelerated(SharedAttentionSettings),
    Unit(SharedAttentionSettings),
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
    attention_storage_types: AttentionStorageTypes,
) -> Result<(), AttentionSetupError> {
    launch_ref(
        strategy,
        client,
        &query.as_ref(),
        &key.as_ref(),
        &value.as_ref(),
        &mask.as_ref().map(|m| m.as_ref()),
        &out.as_ref(),
        attention_storage_types,
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
    attention_storage_types: AttentionStorageTypes,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::BlackboxAccelerated(settings) => {
            launch_attention::<R, BlackboxAcceleratedAlgorithm>(
                client,
                query,
                key,
                value,
                mask,
                out,
                attention_storage_types,
                settings,
            )
        }
        Strategy::Unit(settings) => launch_attention::<R, UnitAlgorithm>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_storage_types,
            settings,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_attention<R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    global_dtypes: AttentionStorageTypes,
    settings: &A::Settings,
) -> Result<(), AttentionSetupError> {
    let line_sizes = {
        let ls = AvailableLineSizes::from_global_types(client, global_dtypes.clone());
        let ls = A::filter_line_sizes(ls)
            .filter_with_tensor(AttentionIdent::Query, query.strides, query.shape)
            .filter_with_tensor(AttentionIdent::Key, key.strides, key.shape)
            .filter_with_tensor(AttentionIdent::Value, value.strides, value.shape)
            .filter_with_tensor(AttentionIdent::Out, out.strides, out.shape);

        if let Some(mask) = mask.as_ref() {
            ls.filter_with_tensor(AttentionIdent::Mask, mask.strides, mask.shape)
        } else {
            ls
        }
    }
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
        line_sizes: line_sizes.clone(),
        global_dtypes,
        accumulator_precision: Default::default(),
    };

    let blueprint = A::blueprint(client, &problem, settings)?;

    let dtypes = A::dtypes(client, &problem, &blueprint)?;

    let cube_count_plan = blueprint.cube_count_plan(&problem);

    let result = unsafe {
        <A as Algorithm>::BatchAttention::launch_unchecked::<TensorArgs, R>(
            client,
            blueprint.cube_dim(),
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
            &dtypes,
            blueprint,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(AttentionSetupError::Execution(err)),
    }
}
