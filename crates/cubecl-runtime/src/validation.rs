use cubecl_common::backtrace::BackTrace;
use cubecl_ir::DeviceProperties;

use crate::{
    id::KernelId,
    server::{CubeDim, LaunchError, ResourceLimitError},
};

/// Validate the cube dim of a kernel fits within the hardware limits
pub fn validate_cube_dim(
    properties: &DeviceProperties,
    kernel_id: &KernelId,
) -> Result<(), LaunchError> {
    let requested = kernel_id.cube_dim;
    let max: CubeDim = properties.hardware.max_cube_dim.into();
    if !max.can_contain(requested) {
        Err(ResourceLimitError::CubeDim {
            requested: requested.into(),
            max: max.into(),
            backtrace: BackTrace::capture(),
        }
        .into())
    } else {
        Ok(())
    }
}

/// Validate the total units of a kernel fits within the hardware limits
pub fn validate_units(
    properties: &DeviceProperties,
    kernel_id: &KernelId,
) -> Result<(), LaunchError> {
    let requested = kernel_id.cube_dim.num_elems();
    let max = properties.hardware.max_units_per_cube;
    if requested > max {
        Err(ResourceLimitError::Units {
            requested,
            max,
            backtrace: BackTrace::capture(),
        }
        .into())
    } else {
        Ok(())
    }
}
