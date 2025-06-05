use cubecl_core::CubeDim;

use crate::components::InvalidConfigError;

pub enum ComputeResources {
    Units(u32),
    Planes(u32),
}

impl ComputeResources {
    pub fn as_plane_resources(self, plane_dim: u32) -> Result<Self, InvalidConfigError> {
        match self {
            ComputeResources::Units(units) => {
                if units % plane_dim == 0 {
                    Ok(ComputeResources::Planes(units / plane_dim))
                } else {
                    Err(Box::new(format!(
                        "Number of units {:?} should be divisible by plane_dim {:?}",
                        units, plane_dim
                    )))
                }
            }
            ComputeResources::Planes(_) => Ok(self),
        }
    }

    pub fn to_cube_dim(self, plane_dim: u32) -> Result<CubeDim, InvalidConfigError> {
        match self {
            ComputeResources::Units(_) => {
                self.as_plane_resources(plane_dim)?.to_cube_dim(plane_dim)
            }
            ComputeResources::Planes(num_planes) => Ok(CubeDim::new_2d(plane_dim, num_planes)),
        }
    }
}
