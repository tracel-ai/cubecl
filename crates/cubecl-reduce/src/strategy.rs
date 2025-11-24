use cubecl_core::prelude::*;
use cubecl_runtime::Plane;
use serde::{Deserialize, Serialize};

use crate::ReduceError;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct ReduceStrategy {
    /// If true and the compute client support plane instructions,
    /// then try using them in the kernel. It could still be impossible to use
    /// plane instructions depending on the memory layout of the tensors.
    pub use_planes: bool,

    /// If true, all units within a single cube cooperate to reduce a single item in the output.
    /// Else, each unit or plane (if planes is true) reduce a single item by itself.
    pub shared: bool,
}

impl ReduceStrategy {
    pub fn validate<R: Runtime>(
        self,
        client: &ComputeClient<R>,
    ) -> Result<Self, ReduceError> {
        if self.use_planes {
            if !support_plane(client) {
                return Err(ReduceError::PlanesUnavailable);
            }
            if !precise_plane_dim(client) {
                return Err(ReduceError::ImprecisePlaneDim);
            }
        }

        Ok(self)
    }

    pub fn new<R: Runtime>(client: &ComputeClient<R>, shared: bool) -> Self {
        Self {
            use_planes: support_plane(client) && precise_plane_dim(client),
            shared,
        }
    }
}

fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}

fn precise_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let hw_props = &client.properties().hardware;
    hw_props.plane_size_min == hw_props.plane_size_max
}
