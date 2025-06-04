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
pub struct LoadingPlaneCount {
    pub overlap: PlaneCountMode,
    pub load_only: PlaneCountMode,
}

impl Default for LoadingPlaneCount {
    fn default() -> Self {
        Self {
            overlap: PlaneCountMode::Inherit,
            load_only: PlaneCountMode::None,
        }
    }
}
impl LoadingPlaneCount {
    pub fn to_plane_roles(&self, compute_planes: u32) -> PlaneRoles {
        let overlap = self.get_overlap_count(compute_planes);
        PlaneRoles {
            load_only: self.load_only.resolve(compute_planes),
            overlap,
            compute_only: compute_planes - overlap,
        }
    }

    fn get_overlap_count(&self, compute_planes: u32) -> u32 {
        let overlap_count = self.overlap.resolve(compute_planes);
        assert!(
            overlap_count <= compute_planes,
            "Overlap count cannot be more than compute planes"
        );
        overlap_count
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlaneCountMode {
    /// Inherit full number of compute planes from the stage matmul.
    Inherit,

    /// Inherit a fractional amount of compute planes.
    InheritFraction { numerator: u32, denominator: u32 },

    /// Fixed number of planes.
    Fixed(u32),

    /// No planes.
    None,
}

impl PlaneCountMode {
    pub fn resolve(&self, compute_planes: u32) -> u32 {
        match *self {
            PlaneCountMode::Inherit => compute_planes,
            PlaneCountMode::InheritFraction {
                numerator,
                denominator,
            } => {
                assert!(
                    numerator <= denominator,
                    "InheritFraction must be between 0 and 1"
                );
                compute_planes * numerator / denominator
            }
            PlaneCountMode::Fixed(n) => n,
            PlaneCountMode::None => 0,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlaneRoles {
    pub load_only: u32,
    pub overlap: u32,
    pub compute_only: u32,
}

impl PlaneRoles {
    pub fn has_specialization(&self) -> bool {
        self.load_only > 0 || self.compute_only > 0
    }

    pub fn loader_count(&self) -> u32 {
        self.load_only + self.overlap
    }

    pub fn computer_count(&self) -> u32 {
        self.compute_only + self.overlap
    }
}
