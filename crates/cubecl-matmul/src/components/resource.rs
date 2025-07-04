use cubecl_core::prelude::*;

use crate::{
    components::{
        InvalidConfigError,
        global::{MaxLoaders, PlaneRoles},
    },
    gcd,
};

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
                        "Number of units {units:?} should be divisible by plane_dim {plane_dim:?}"
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

/// Configuration for how each input tensor (LHS and RHS) is loaded,
/// specifying the plane roles responsible for loading them.
#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoadSpecializationConfig {
    /// Load strategy for the LHS (left-hand side) tensor.
    pub lhs: SpecializationTensorConfig,
    /// Load strategy for the RHS (right-hand side) tensor.
    pub rhs: SpecializationTensorConfig,
}

/// Determines which types of planes are responsible for loading a tensor.
///
/// TODO: maybe we want a "MainPlusExtra" variant that uses main flow planes and load-only planes
/// for the same tensor
#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SpecializationTensorConfig {
    /// The tensor is loaded exclusively by planes that participate in the main computation flow.
    #[default]
    MainFlowOnly,

    /// The tensor is loaded exclusively by planes dedicated to loading (load-only planes),
    /// which do not participate in computation.
    LoadFlowOnly,
}

impl LoadSpecializationConfig {
    pub fn has_specialization(&self) -> bool {
        self.lhs.has_specialization() || self.rhs.has_specialization()
    }
}

impl SpecializationTensorConfig {
    pub fn has_specialization(&self) -> bool {
        match self {
            SpecializationTensorConfig::MainFlowOnly => false,
            SpecializationTensorConfig::LoadFlowOnly => true,
        }
    }
}

impl LoadSpecializationConfig {
    pub fn to_plane_roles(&self, main_flow: u32, loader_tasks: MaxLoaders) -> PlaneRoles {
        PlaneRoles {
            main_flow,
            load_only: self.find_num_load_only(main_flow, loader_tasks),
        }
    }

    fn find_num_load_only(&self, main_flow: u32, loader_tasks: MaxLoaders) -> u32 {
        use SpecializationTensorConfig::*;

        let ideal_load_only = match (self.lhs, self.rhs) {
            (MainFlowOnly, MainFlowOnly) => 0,
            (MainFlowOnly, LoadFlowOnly) => loader_tasks.rhs,
            (LoadFlowOnly, MainFlowOnly) => loader_tasks.lhs,
            (LoadFlowOnly, LoadFlowOnly) => gcd(loader_tasks.lhs, loader_tasks.rhs),
        };

        // Don't stray too far from main_flow
        best_divisor_close_to_reference(ideal_load_only, main_flow)
    }
}

fn best_divisor_close_to_reference(dividible_value: u32, reference: u32) -> u32 {
    let mut best = 1;
    let mut best_dist = reference.abs_diff(1);

    for d in 1..=dividible_value {
        if dividible_value % d == 0 {
            let dist = reference.abs_diff(d);
            if dist < best_dist || (dist == best_dist && d > best) {
                best = d;
                best_dist = dist;
            }
        }
    }

    best
}
