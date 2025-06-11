use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::InputIdent;

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
pub enum LoadingSides {
    // Load both Lhs and Rhs
    Both,
    // Load Lhs only
    Lhs,
    // Load Rhs only
    Rhs,
    // Don't perform loading
    None,
}

impl LoadingSides {
    pub fn includes_lhs(&self) -> bool {
        self.includes(InputIdent::Lhs)
    }

    pub fn includes_rhs(&self) -> bool {
        self.includes(InputIdent::Rhs)
    }

    pub fn includes(&self, ident: InputIdent) -> bool {
        matches!(
            (self, ident),
            (LoadingSides::Both, _)
                | (LoadingSides::Lhs, InputIdent::Lhs)
                | (LoadingSides::Rhs, InputIdent::Rhs)
        )
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SpecializedLoadingSides {
    pub main_flow: LoadingSides,
    pub load_only: LoadingSides,
}

impl SpecializedLoadingSides {
    pub fn num_loading_planes(
        &self,
        specialized: bool,
        ident: InputIdent,
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
pub enum RoleRule {
    MainFlowOnly,
    // Load-only planes are first, then come main flow
    LoadOnlyFirst(u32),
    // Main flow planes are first, then come load-only
    LoadOnlyLast(u32),
}

impl PlaneRoleConfig {
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
            RoleRuleConfig::LoadOnlyFirst { load_only } => RoleRule::new_LoadOnlyFirst(load_only),
            RoleRuleConfig::LoadOnlyLast { main_flow } => RoleRule::new_LoadOnlyLast(main_flow),
        }
    }

    pub fn is_load_only(self) -> bool {
        match self {
            RoleRule::MainFlowOnly => false,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y < load_only,
            RoleRule::LoadOnlyLast(main_flow) => UNIT_POS_Y >= main_flow,
        }
    }

    pub fn compute_index(self) -> u32 {
        match self {
            RoleRule::MainFlowOnly => UNIT_POS_Y,
            RoleRule::LoadOnlyFirst(load_only) => UNIT_POS_Y - load_only,
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
                if !specialized_loading_sides.load_only.includes(ident) {
                    UNIT_POS_Y - load_only
                } else {
                    UNIT_POS_Y
                }
            }
            RoleRule::LoadOnlyLast(main_flow) => {
                if !specialized_loading_sides.main_flow.includes(ident) {
                    UNIT_POS_Y - main_flow
                } else {
                    UNIT_POS_Y
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SpecializerKind {
    Specialized {
        main_flow_loading_side: LoadingSides,
        load_only_loading_side: LoadingSides,
        role_rule_config: RoleRuleConfig,
    },
    NotSpecialized,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Specializer {
    #[cube(comptime)]
    pub kind: SpecializerKind,
}

#[cube]
impl Specializer {
    pub fn new(
        #[comptime] plane_role_config: PlaneRoleConfig,
        #[comptime] loading_sides: SpecializedLoadingSides,
    ) -> Specializer {
        if plane_role_config.has_specialization() {
            Specializer {
                kind: comptime! {
                    SpecializerKind::Specialized {
                        main_flow_loading_side: loading_sides.main_flow,
                        load_only_loading_side: loading_sides.load_only,
                        role_rule_config: plane_role_config.rule
                    }
                },
            }
        } else {
            Specializer {
                kind: comptime! {SpecializerKind::NotSpecialized},
            }
        }
    }
}
