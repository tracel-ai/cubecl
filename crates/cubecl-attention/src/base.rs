use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::components::{AttentionPrecision, AttentionSetupError};

use super::kernels::tmp;

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
        Strategy::Tmp => {
            tmp::launch_ref::<R>(client, query, key, value, out)?;
        }
    }

    Ok(())
}
