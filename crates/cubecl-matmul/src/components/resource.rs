use cubecl_core::prelude::*;

use crate::components::InvalidConfigError;

/// Number of compute primitives required by some component, specified as either units or planes.
pub enum ComputeResources {
    Units(u32),
    Planes(u32),
}

impl ComputeResources {
    /// Ensures [ComputeResources] is Planes variant, converting
    /// units using plane_dim, the number of units in a plane.
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub fn as_plane_resources(self, plane_dim: u32) -> Result<Self, InvalidConfigError> {
        match self {
            ComputeResources::Units(units) => {
                if units % plane_dim == 0 {
                    Ok(ComputeResources::Planes(units / plane_dim))
                } else {
                    Err(Box::new(format!(
                        "Number of units {units:?} should be divisible by plane_dim {plane_dim:?}"
                    )))
                }
            }
            ComputeResources::Planes(_) => Ok(self),
        }
    }

    /// Make a [CubeDim] from specified resources.
    ///
    /// Obtained CubeDim is always (plane_dim, number_of_planes, 1)
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub fn to_cube_dim(self, plane_dim: u32) -> Result<CubeDim, InvalidConfigError> {
        match self {
            ComputeResources::Units(_) => {
                self.as_plane_resources(plane_dim)?.to_cube_dim(plane_dim)
            }
            ComputeResources::Planes(num_planes) => Ok(CubeDim::new_2d(plane_dim, num_planes)),
        }
    }

    /// Get the number of planes
    ///
    /// Will fail if the number of units does not correspond to an exact number of planes
    pub(crate) fn num_planes(self, plane_dim: u32) -> Result<u32, InvalidConfigError> {
        let plane_resources = self.as_plane_resources(plane_dim)?;
        if let ComputeResources::Planes(num_planes) = plane_resources {
            Ok(num_planes)
        } else {
            unreachable!()
        }
    }
}
