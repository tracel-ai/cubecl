use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::global::{GlobalConfig, LoaderTasksMap};
use crate::components::tile::TileConfig;
use crate::components::{InputIdent, LoadSpecializationConfig, SpecializationTensorConfig};
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
    pub fn new<T: TileConfig>(
        load_specialization_config: LoadSpecializationConfig,
        loader_tasks_map: Option<LoaderTasksMap>,
        num_main_flow_planes: u32,
        tile_config: &T,
    ) -> Result<PlaneRoleConfig, MatmulSetupError> {
        let plane_roles = match loader_tasks_map {
            Some(loader_tasks_map) => load_specialization_config.to_plane_roles(
                num_main_flow_planes,
                loader_tasks_map.resolve(
                    tile_config.global_line_size(InputIdent::Lhs) as u8,
                    tile_config.global_line_size(InputIdent::Rhs) as u8,
                ),
            ),

            None => {
                if load_specialization_config.has_specialization() {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(
                        "Error: Load specialization config has specialization but no loader tasks map was given."
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
    pub fn new<G: GlobalConfig>(#[comptime] config: G) -> Specializer {
        let plane_role_config = config.plane_role_config();
        let loading_sides = config.specialized_loading_sides();

        if config.plane_role_config().has_specialization() {
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
