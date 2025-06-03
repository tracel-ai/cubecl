use cubecl_core::CubeDim;

use crate::components::InvalidConfigError;

pub enum ResourceDemand {
    Units(u32),
    Planes(u32),
}

impl ResourceDemand {
    pub fn as_planes_resource(self, plane_dim: u32) -> Result<Self, InvalidConfigError> {
        match self {
            ResourceDemand::Units(units) => {
                if units % plane_dim == 0 {
                    Ok(ResourceDemand::Planes(units / plane_dim))
                } else {
                    Err(Box::new(format!(
                        "Number of units {:?} should be divisible by plane_dim {:?}",
                        units, plane_dim
                    )))
                }
            }
            ResourceDemand::Planes(_) => Ok(self),
        }
    }

    pub fn to_cube_dim(self, plane_dim: u32) -> Result<CubeDim, InvalidConfigError> {
        match self {
            ResourceDemand::Units(_) => self.as_planes_resource(plane_dim)?.to_cube_dim(plane_dim),
            ResourceDemand::Planes(num_planes) => Ok(CubeDim::new_2d(plane_dim, num_planes)),
        }
    }
}
