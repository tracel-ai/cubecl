use crate::components::{LoadOnlyRoleConfig, PlaneRoles};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SpecializerConfig {
    plane_roles: PlaneRoles,
    kind: SpecializerKind,
}

// remains comptime. is use inside specializer AND to create specializer
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SpecializerKind {
    LoadOnlyFirst,
    NoSpecialization,
}

impl SpecializerConfig {
    pub fn from_plane_roles_config(
        plane_role_config: LoadOnlyRoleConfig,
        compute_planes: u32,
    ) -> Self {
        Self::from_plane_roles(plane_role_config.to_plane_roles(compute_planes))
    }

    pub fn from_plane_roles(plane_roles: PlaneRoles) -> Self {
        let kind = match plane_roles.has_specialization() {
            true => SpecializerKind::LoadOnlyFirst,
            false => SpecializerKind::NoSpecialization,
        };

        Self { plane_roles, kind }
    }

    pub fn loader_count(&self) -> u32 {
        self.plane_roles.loader_count()
    }

    pub fn computer_count(&self) -> u32 {
        self.plane_roles.computer_count()
    }

    fn must_check_if_computer(&self) -> bool {
        match self.kind {
            SpecializerKind::LoadOnlyFirst => self.plane_roles.load_only > 0,
            SpecializerKind::NoSpecialization => false,
        }
    }

    pub fn has_specialization(&self) -> bool {
        self.plane_roles.has_specialization()
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Specializer {
    #[cube(comptime)]
    specializer_config: SpecializerConfig,
}

#[cube]
impl Specializer {
    pub fn new(#[comptime] specializer_config: SpecializerConfig) -> Specializer {
        Specializer { specializer_config }
    }

    pub fn can_be_load_only(&self) -> comptime_type!(bool) {
        comptime!(self.specializer_config.must_check_if_computer())
    }

    pub fn is_computer(&self) -> bool {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOnlyFirst => {
                UNIT_POS_Y >= self.specializer_config.plane_roles.load_only
            }
            SpecializerKind::NoSpecialization => {
                comptime!(unreachable!("Should call must_check_if_computer prior"))
            }
        }
    }

    pub fn plane_id_to_loader_index(&self) -> u32 {
        UNIT_POS_Y
    }

    pub fn plane_id_to_computer_index(&self) -> u32 {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOnlyFirst => {
                UNIT_POS_Y - self.specializer_config.plane_roles.load_only
            }
            SpecializerKind::NoSpecialization => UNIT_POS_Y,
        }
    }
}
