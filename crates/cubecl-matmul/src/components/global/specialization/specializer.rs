use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::global::GlobalConfig;
use crate::components::global::specialization::config::LoadingSides;
use crate::components::global::specialization::roles::RoleRuleConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Comptime information of specializer
pub enum SpecializerKind {
    Specialized {
        main_flow_loading_side: LoadingSides,
        load_only_loading_side: LoadingSides,
        role_rule_config: RoleRuleConfig,
    },
    NotSpecialized,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Specialization information in cube functions
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
