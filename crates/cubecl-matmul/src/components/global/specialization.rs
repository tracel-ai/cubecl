use crate::components::PlaneRoles;
use cubecl_core::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ThresholdSpecializer {
    #[cube(comptime)]
    pub loader_end: u32,
    #[cube(comptime)]
    pub compute_start: u32,
}

#[cube]
impl ThresholdSpecializer {
    pub fn new(#[comptime] plane_roles: PlaneRoles) -> Self {
        ThresholdSpecializer {
            loader_end: plane_roles.load_only + plane_roles.overlap,
            compute_start: plane_roles.load_only,
        }
    }

    pub fn is_loader(&self, plane_id: u32) -> bool {
        plane_id < self.loader_end
    }

    pub fn is_computer(&self, plane_id: u32) -> bool {
        plane_id >= self.compute_start
    }

    pub fn plane_id_to_loader_index(&self, plane_id: u32) -> u32 {
        plane_id
    }

    pub fn plane_id_to_computer_index(&self, plane_id: u32) -> u32 {
        plane_id - self.compute_start
    }

    pub fn loader_index_to_plane_id(&self, loader_index: u32) -> u32 {
        loader_index
    }

    pub fn computer_index_to_plane_id(&self, computer_index: u32) -> u32 {
        self.compute_start + computer_index
    }
}
