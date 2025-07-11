//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::AttentionSetupError;

#[cube(launch_unchecked)]
fn bad_attention<N: Numeric>(
    query: &Tensor<Line<N>>,
    key: &Tensor<Line<N>>,
    value: &Tensor<Line<N>>,
    out: &mut Tensor<Line<N>>,
) {
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), AttentionSetupError> {
    todo!()
}
