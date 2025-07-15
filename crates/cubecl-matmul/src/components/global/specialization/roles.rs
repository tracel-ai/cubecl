use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::InputIdent;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaderPlanes;
use crate::components::global::specialization::config::{
    LoadSpecializationConfig, SpecializedLoadingSides,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Represents how many planes are used for main matmul computation and for loading-only tasks.
pub struct PlaneRoles {
    /// Number of planes participating in main matmul and (possibly) loading.
    pub main_flow: u32,
    /// Number of planes dedicated solely to loading.
    pub load_only: u32,
}

impl PlaneRoles {
    /// Return the total number of planes
    pub fn total_count(&self) -> u32 {
        self.main_flow + self.load_only
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Contains the number of plane in each role and the rule to distinguish planes based on their plane id
pub struct PlaneRoleConfig {
    pub plane_roles: PlaneRoles,
    pub rule: RoleRuleConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Comptime version of [RoleRule]
pub enum RoleRuleConfig {
    MainFlowOnly,
    LoadOnlyFirst { load_only: u32 },
    LoadOnlyLast { main_flow: u32 },
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Threshold of plane id at which the roles change
///
/// Note: this struct is only necessary because Cube enums cannot hold
/// a comptime value directly
pub struct Threshold {
    #[cube(comptime)]
    threshold: u32,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Rule to distinguish a plane's role based on its plane id
pub enum RoleRule {
    /// All planes are in the main flow, this is equivalent of having no specialization
    MainFlowOnly,
    /// Load-only planes: [0, Threshold)
    /// Main flow planes: [Threshold, total)
    LoadOnlyFirst(Threshold),
    /// Main flow planes: [0, Threshold)
    /// Load-only planes: [Threshold, total)
    LoadOnlyLast(Threshold),
}

impl PlaneRoleConfig {
    /// Make a new PlaneRoleConfig
    pub fn new(
        load_specialization_config: LoadSpecializationConfig,
        loader_tasks: Option<MaxLoaderPlanes>,
        num_main_flow_planes: u32,
    ) -> Result<PlaneRoleConfig, MatmulSetupError> {
        let plane_roles = match loader_tasks {
            Some(loader_tasks) => {
                load_specialization_config.to_plane_roles(num_main_flow_planes, loader_tasks)
            }

            None => {
                if load_specialization_config.has_specialization() {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(
                        "Error: Load specialization config has specialization but no loader tasks were given."
                            .to_string(),
                    )));
                } else {
                    PlaneRoles {
                        main_flow: num_main_flow_planes,
                        load_only: 0,
                    }
                }
            }
        };

        // TODO make possible to select LoadOnlyLast
        let rule = match plane_roles.load_only {
            0 => RoleRuleConfig::MainFlowOnly,
            _ => RoleRuleConfig::LoadOnlyFirst {
                load_only: plane_roles.load_only,
            },
        };

        Ok(Self { plane_roles, rule })
    }

    /// Returns the number of planes participating in main flow
    pub fn main_flow_count(&self) -> u32 {
        self.plane_roles.main_flow
    }

    /// Whether the plane role config implies specialization
    pub fn has_specialization(&self) -> bool {
        self.plane_roles.load_only > 0
    }
}

#[cube]
impl RoleRule {
    /// Make a cube role rule from comptime config
    pub fn new(#[comptime] comptime_rule: RoleRuleConfig) -> RoleRule {
        match comptime!(comptime_rule) {
            RoleRuleConfig::MainFlowOnly => RoleRule::new_MainFlowOnly(),
            RoleRuleConfig::LoadOnlyFirst { load_only } => RoleRule::new_LoadOnlyFirst(Threshold {
                threshold: load_only,
            }),
            RoleRuleConfig::LoadOnlyLast { main_flow } => RoleRule::new_LoadOnlyLast(Threshold {
                threshold: main_flow,
            }),
        }
    }

    /// Whether the current plane is a load-only plane
    pub fn is_load_only(self) -> bool {
        match self {
            RoleRule::MainFlowOnly => false,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y < load_only.threshold,
            RoleRule::LoadOnlyLast(main_flow) => UNIT_POS_Y >= main_flow.threshold,
        }
    }

    /// The index of the current plane among planes that perform compute,
    /// ignoring load-only planes
    pub fn compute_index(self) -> u32 {
        match self {
            RoleRule::MainFlowOnly => UNIT_POS_Y,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y - load_only.threshold,
            RoleRule::LoadOnlyLast(_) => UNIT_POS_Y,
        }
    }

    /// The index of the current plane among planes that perform loading,
    /// ignoring any plane that does not participate for this `ident`.
    pub fn load_index(
        self,
        #[comptime] ident: InputIdent,
        #[comptime] specialized_loading_sides: SpecializedLoadingSides,
    ) -> u32 {
        match self {
            RoleRule::MainFlowOnly => UNIT_POS_Y,
            RoleRule::LoadOnlyFirst(load_only) => {
                if comptime!(!specialized_loading_sides.load_only.includes(ident)) {
                    UNIT_POS_Y - load_only.threshold
                } else {
                    UNIT_POS_Y
                }
            }
            RoleRule::LoadOnlyLast(main_flow) => {
                if comptime!(!specialized_loading_sides.main_flow.includes(ident)) {
                    UNIT_POS_Y - main_flow.threshold
                } else {
                    UNIT_POS_Y
                }
            }
        }
    }
}
