use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::InputIdent;
use crate::components::global::MaxLoaders;
use crate::components::global::specialization::config::{
    LoadSpecializationConfig, SpecializedLoadingSides,
};
use crate::kernels::MatmulSetupError;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlaneRoles {
    pub main_flow: u32,
    pub load_only: u32,
}

impl PlaneRoles {
    pub fn total_count(&self) -> u32 {
        self.main_flow + self.load_only
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlaneRoleConfig {
    pub plane_roles: PlaneRoles,
    pub rule: RoleRuleConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RoleRuleConfig {
    MainFlowOnly,
    LoadOnlyFirst { load_only: u32 },
    LoadOnlyLast { main_flow: u32 },
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Threshold {
    #[cube(comptime)]
    threshold: u32,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RoleRule {
    MainFlowOnly,
    // Load-only planes are first, then come main flow
    LoadOnlyFirst(Threshold),
    // Main flow planes are first, then come load-only
    LoadOnlyLast(Threshold),
}

impl PlaneRoleConfig {
    pub fn new(
        load_specialization_config: LoadSpecializationConfig,
        loader_tasks: Option<MaxLoaders>,
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

        Ok(Self::from_plane_roles(plane_roles))
    }

    pub fn from_plane_roles(plane_roles: PlaneRoles) -> Self {
        // TODO make possible to select LoadOnlyLast
        let rule = match plane_roles.load_only {
            0 => RoleRuleConfig::MainFlowOnly,
            _ => RoleRuleConfig::LoadOnlyFirst {
                load_only: plane_roles.load_only,
            },
        };

        Self { plane_roles, rule }
    }

    pub fn main_flow_count(&self) -> u32 {
        self.plane_roles.main_flow
    }

    pub fn has_specialization(&self) -> bool {
        self.plane_roles.load_only > 0
    }
}

#[cube]
impl RoleRule {
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

    pub fn is_load_only(self) -> bool {
        match self {
            RoleRule::MainFlowOnly => false,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y < load_only.threshold,
            RoleRule::LoadOnlyLast(main_flow) => UNIT_POS_Y >= main_flow.threshold,
        }
    }

    pub fn compute_index(self) -> u32 {
        match self {
            RoleRule::MainFlowOnly => UNIT_POS_Y,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y - load_only.threshold,
            RoleRule::LoadOnlyLast(_) => UNIT_POS_Y,
        }
    }

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
