use crate::components::PlaneRoles;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait SpecializationRule: CubeType {
    /// Returns true if the given plane_id is a loader
    fn is_loader(this: &Self, plane_id: u32) -> bool;

    /// Returns true if the given plane_id is a computer
    fn is_computer(this: &Self, plane_id: u32) -> bool;

    /// Maps a global plane ID to its loader index (0-based)
    fn plane_id_to_loader_index(this: &Self, plane_id: u32) -> u32;

    /// Maps a global plane ID to its computer index (0-based)
    fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32;

    /// Maps a loader index (0-based) to the global plane ID.
    fn loader_index_to_plane_id(this: &Self, loader_index: u32) -> u32;

    /// Maps a computer index (0-based) to the global plane ID.
    fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32;
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ThresholdSpecializer {
    pub loader_end: u32,
    pub compute_start: u32,
}

#[cube]
impl SpecializationRule for ThresholdSpecializer {
    fn is_loader(this: &Self, plane_id: u32) -> bool {
        plane_id < this.loader_end
    }

    fn is_computer(this: &Self, plane_id: u32) -> bool {
        plane_id >= this.compute_start
    }

    fn plane_id_to_loader_index(_this: &Self, plane_id: u32) -> u32 {
        plane_id
    }

    fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32 {
        plane_id - this.compute_start
    }

    fn loader_index_to_plane_id(_this: &Self, loader_index: u32) -> u32 {
        loader_index
    }

    fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32 {
        this.compute_start + computer_index
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Specializer {
    NoSpecialization,
    Threshold(ThresholdSpecializer),
}

#[cube]
impl SpecializationRule for Specializer {
    fn is_loader(this: &Self, plane_id: u32) -> bool {
        match this {
            Specializer::NoSpecialization => true,
            Specializer::Threshold(s) => ThresholdSpecializer::is_loader(s, plane_id),
        }
    }

    fn is_computer(this: &Self, plane_id: u32) -> bool {
        match this {
            Specializer::NoSpecialization => true,
            Specializer::Threshold(s) => ThresholdSpecializer::is_computer(s, plane_id),
        }
    }

    fn plane_id_to_loader_index(this: &Self, plane_id: u32) -> u32 {
        match this {
            Specializer::NoSpecialization => plane_id,
            Specializer::Threshold(s) => {
                ThresholdSpecializer::plane_id_to_loader_index(s, plane_id)
            }
        }
    }

    fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32 {
        match this {
            Specializer::NoSpecialization => plane_id,
            Specializer::Threshold(s) => {
                ThresholdSpecializer::plane_id_to_computer_index(s, plane_id)
            }
        }
    }

    fn loader_index_to_plane_id(this: &Self, loader_index: u32) -> u32 {
        match this {
            Specializer::NoSpecialization => loader_index,
            Specializer::Threshold(s) => {
                ThresholdSpecializer::loader_index_to_plane_id(s, loader_index)
            }
        }
    }

    fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32 {
        match this {
            Specializer::NoSpecialization => computer_index,
            Specializer::Threshold(s) => {
                ThresholdSpecializer::computer_index_to_plane_id(s, computer_index)
            }
        }
    }
}

pub fn new_specializer(plane_roles: PlaneRoles) -> Specializer {
    if plane_roles.has_specialization() {
        Specializer::Threshold(ThresholdSpecializer {
            loader_end: plane_roles.load_only + plane_roles.overlap,
            compute_start: plane_roles.load_only,
        })
    } else {
        Specializer::NoSpecialization
    }
}

#[cube]
impl Specializer {
    pub fn is_loader(&self) -> bool {
        <Specializer as SpecializationRule>::is_loader(self, UNIT_POS_Y)
    }

    pub fn is_computer(&self) -> bool {
        <Specializer as SpecializationRule>::is_computer(self, UNIT_POS_Y)
    }

    pub fn plane_id_to_loader_index(&self) -> u32 {
        <Specializer as SpecializationRule>::plane_id_to_loader_index(self, UNIT_POS_Y)
    }

    pub fn plane_id_to_computer_index(&self) -> u32 {
        <Specializer as SpecializationRule>::plane_id_to_computer_index(self, UNIT_POS_Y)
    }

    pub fn loader_index_to_plane_id(&self, loader_index: u32) -> u32 {
        <Specializer as SpecializationRule>::loader_index_to_plane_id(self, loader_index)
    }

    pub fn computer_index_to_plane_id(&self, computer_index: u32) -> u32 {
        <Specializer as SpecializationRule>::computer_index_to_plane_id(self, computer_index)
    }
}
