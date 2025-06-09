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

    pub(crate) fn get_count(&self) -> u32 {
        match self {
            ComputeResources::Units(count) => *count,
            ComputeResources::Planes(count) => *count,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadOnlyRoleConfig {
    /// Use the number of compute planes from the stage matmul.
    Mirror,

    /// Use a fractional amount of compute planes.
    MirrorRatio { numerator: u32, denominator: u32 },

    /// Fixed number of planes.
    Fixed(u32),

    /// No planes.
    None,
}

impl LoadOnlyRoleConfig {
    pub fn to_plane_roles(&self, main_flow: u32) -> PlaneRoles {
        PlaneRoles {
            main_flow,
            load_only: self.resolve(main_flow),
        }
    }

    pub fn resolve(&self, main_flow_planes: u32) -> u32 {
        match *self {
            Self::Mirror => main_flow_planes,
            Self::MirrorRatio {
                numerator,
                denominator,
            } => {
                assert!(
                    numerator <= denominator,
                    "MirrorRatio must be between 0 and 1"
                );
                main_flow_planes * numerator / denominator
            }
            Self::Fixed(n) => n,
            Self::None => 0,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlaneRoles {
    pub main_flow: u32,
    pub load_only: u32,
}

impl PlaneRoles {
    pub fn has_specialization(&self) -> bool {
        self.load_only > 0
    }

    pub fn loader_count(&self) -> u32 {
        self.load_only + self.main_flow
    }

    pub fn computer_count(&self) -> u32 {
        self.main_flow
    }
}
