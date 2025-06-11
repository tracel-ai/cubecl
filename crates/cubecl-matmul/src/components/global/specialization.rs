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
pub enum LoadingSet {
    // Load both Lhs and Rhs
    Full,
    // Load Lhs only
    Lhs,
    // Load Rhs only
    Rhs,
    // Don't perform loading
    None,
}

impl LoadingSet {
    pub fn should_fill_lhs(&self) -> bool {
        match self {
            LoadingSet::Full => true,
            LoadingSet::Lhs => true,
            LoadingSet::Rhs => false,
            LoadingSet::None => false,
        }
    }

    pub fn should_fill_rhs(&self) -> bool {
        match self {
            LoadingSet::Full => true,
            LoadingSet::Lhs => false,
            LoadingSet::Rhs => true,
            LoadingSet::None => false,
        }
    }

    pub fn includes(&self, ident: InputIdent) -> bool {
        match (self, ident) {
            (LoadingSet::Full, _)
            | (LoadingSet::Lhs, InputIdent::Lhs)
            | (LoadingSet::Rhs, InputIdent::Rhs) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoadingSets {
    pub specialized_main_flow: LoadingSet,
    pub specialized_load_only: LoadingSet,
    // TODO it's always full
    pub no_specialization: LoadingSet,
}

impl LoadingSets {
    pub fn num_loading_planes(
        &self,
        specialized: bool,
        ident: InputIdent,
        plane_roles: PlaneRoles,
    ) -> u32 {
        let mut num_loading_planes = 0;

        if specialized {
            if self.specialized_main_flow.includes(ident) {
                num_loading_planes += plane_roles.main_flow;
            }
            if self.specialized_load_only.includes(ident) {
                num_loading_planes += plane_roles.load_only;
            }
        } else {
            if self.no_specialization.includes(ident) {
                num_loading_planes += plane_roles.main_flow;
            }
        }

        num_loading_planes
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

    pub fn loader_count(&self) -> u32 {
        self.plane_roles.load_only + self.plane_roles.main_flow
    }

    pub fn load_only_count(&self) -> u32 {
        self.plane_roles.load_only
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
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SpecializerKind {
    Specialized {
        main_flow_loading_set: LoadingSet,
        load_only_loading_set: LoadingSet,
        role_rule_config: RoleRuleConfig,
    },
    NotSpecialized(LoadingSet),
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
        #[comptime] loading_sets: LoadingSets,
    ) -> Specializer {
        if plane_role_config.has_specialization() {
            Specializer {
                kind: comptime! {
                    SpecializerKind::Specialized {
                        main_flow_loading_set: loading_sets.specialized_main_flow,
                        load_only_loading_set: loading_sets.specialized_load_only,
                        role_rule_config: plane_role_config.rule
                    }
                },
            }
        } else {
            Specializer {
                kind: comptime! {SpecializerKind::NotSpecialized(
                    loading_sets.no_specialization,
                )},
            }
        }
    }
}
