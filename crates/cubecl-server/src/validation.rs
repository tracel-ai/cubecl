use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::{DeviceProperties, settings::Dim3};

use crate::{
    id::KernelId,
    server::{LaunchError, ResourceLimitError},
};

/// Validate the cube dim of a kernel fits within the hardware limits
pub fn validate_cube_dim(
    properties: &DeviceProperties,
    kernel_id: &KernelId,
) -> Result<(), LaunchError> {
    let requested = kernel_id.cube_dim;
    let max: Dim3 = properties.hardware.max_cube_dim.into();
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

/// Validate the shared memory a compiled kernel asks for fits within the
/// hardware limits. `requested` is `None` for a kernel with nothing to read
/// it from — precompiled text declares its shared memory statically.
pub fn validate_shared_memory(
    properties: &DeviceProperties,
    requested: Option<usize>,
) -> Result<(), LaunchError> {
    let max = properties.hardware.max_shared_memory_size;
    match requested {
        Some(requested) if requested > max => Err(ResourceLimitError::SharedMemory {
            requested,
            max,
            backtrace: BackTrace::capture(),
        }
        .into()),
        _ => Ok(()),
    }
}
