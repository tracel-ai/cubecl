use crate::components::{
    MatmulIdent,
    global::{MaxGlobalReaderPlanes, specialization::roles::PlaneRoles},
};

/// Configuration for how each input tensor (Lhs and Rhs) is loaded,
/// specifying the plane roles responsible for loading them.
#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoadSpecializationConfig {
    /// Load strategy for the Lhs tensor.
    pub lhs: SpecializationTensorConfig,
    /// Load strategy for the Rhs tensor.
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
    /// Whether there is specialization in the algorithm
    pub fn has_specialization(&self) -> bool {
        self.lhs.has_specialization() || self.rhs.has_specialization()
    }
}

impl SpecializationTensorConfig {
    /// Whether there is specialization for the tensor
    pub fn has_specialization(&self) -> bool {
        match self {
            SpecializationTensorConfig::MainFlowOnly => false,
            SpecializationTensorConfig::LoadFlowOnly => true,
        }
    }
}

impl LoadSpecializationConfig {
    /// Computes how many planes of each role there should be,
    /// using the number of planes needed for main execution, and how
    /// many planes each reader can handle
    ///
    /// The strategy is to find a balanced divisor for reader planes that stays as
    /// close as possible to the main execution plane count.
    pub fn to_plane_roles(
        &self,
        main_flow: u32,
        reader_tasks: MaxGlobalReaderPlanes,
    ) -> PlaneRoles {
        use SpecializationTensorConfig::*;

        let ideal_load_only = match (self.lhs, self.rhs) {
            (MainFlowOnly, MainFlowOnly) => 0,
            (MainFlowOnly, LoadFlowOnly) => reader_tasks.rhs,
            (LoadFlowOnly, MainFlowOnly) => reader_tasks.lhs,
            (LoadFlowOnly, LoadFlowOnly) => gcd(reader_tasks.lhs, reader_tasks.rhs),
        };

        // Don't stray too far from main_flow
        let load_only = best_divisor_close_to_reference(ideal_load_only, main_flow);

        PlaneRoles {
            main_flow,
            load_only,
        }
    }
}

/// Returns the divisor of `dividible_value` closest to `reference`, preferring larger on ties.
fn best_divisor_close_to_reference(dividible_value: u32, reference: u32) -> u32 {
    let mut best = 1;
    let mut best_dist = reference.abs_diff(1);

    for d in 1..=dividible_value {
        if dividible_value.is_multiple_of(d) {
            let dist = reference.abs_diff(d);
            if dist < best_dist || (dist == best_dist && d > best) {
                best = d;
                best_dist = dist;
            }
        }
    }

    best
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Specifies which input(s) a plane role participates in loading.
pub enum LoadingSides {
    /// Load both Lhs and Rhs
    Both,
    /// Load Lhs only
    Lhs,
    /// Load Rhs only
    Rhs,
    /// Don't perform loading
    None,
}

impl LoadingSides {
    /// Returns `true` if Lhs is included.
    pub fn includes_lhs(&self) -> bool {
        self.includes(MatmulIdent::Lhs)
    }

    /// Returns `true` if Rhs is included.
    pub fn includes_rhs(&self) -> bool {
        self.includes(MatmulIdent::Rhs)
    }

    /// Returns `true` if the given input is included.
    pub fn includes(&self, ident: MatmulIdent) -> bool {
        matches!(
            (self, ident),
            (LoadingSides::Both, _)
                | (LoadingSides::Lhs, MatmulIdent::Lhs)
                | (LoadingSides::Rhs, MatmulIdent::Rhs)
        )
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Aggregates loading sides for both main flow and load only roles
pub struct SpecializedLoadingSides {
    pub main_flow: LoadingSides,
    pub load_only: LoadingSides,
}

impl SpecializedLoadingSides {
    /// Returns the number of planes participating in the loading of `ident`
    pub fn num_loading_planes(
        &self,
        specialized: bool,
        ident: MatmulIdent,
        plane_roles: PlaneRoles,
    ) -> u32 {
        if specialized {
            let mut num_loading_planes = 0;
            if self.main_flow.includes(ident) {
                num_loading_planes += plane_roles.main_flow;
            }
            if self.load_only.includes(ident) {
                num_loading_planes += plane_roles.load_only;
            }
            num_loading_planes
        } else {
            plane_roles.main_flow
        }
    }
}

impl From<LoadSpecializationConfig> for SpecializedLoadingSides {
    fn from(lsc: LoadSpecializationConfig) -> Self {
        use SpecializationTensorConfig::*;
        match (lsc.lhs, lsc.rhs) {
            (MainFlowOnly, MainFlowOnly) => SpecializedLoadingSides {
                main_flow: LoadingSides::Both,
                load_only: LoadingSides::None,
            },
            (MainFlowOnly, LoadFlowOnly) => SpecializedLoadingSides {
                main_flow: LoadingSides::Lhs,
                load_only: LoadingSides::Rhs,
            },
            (LoadFlowOnly, MainFlowOnly) => SpecializedLoadingSides {
                main_flow: LoadingSides::Rhs,
                load_only: LoadingSides::Lhs,
            },
            (LoadFlowOnly, LoadFlowOnly) => SpecializedLoadingSides {
                main_flow: LoadingSides::None,
                load_only: LoadingSides::Both,
            },
        }
    }
}

pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
