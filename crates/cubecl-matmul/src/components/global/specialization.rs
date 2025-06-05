use crate::components::{LoadingPlaneCount, PlaneRoles};
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
    LoadOverlapCompute,
    NoSpecialization,
}

impl SpecializerConfig {
    pub fn from_loading_plane_count(
        loading_plane_count: LoadingPlaneCount,
        compute_planes: u32,
    ) -> Self {
        Self::from_plane_roles(loading_plane_count.to_plane_roles(compute_planes))
    }

    pub fn from_plane_roles(plane_roles: PlaneRoles) -> Self {
        let kind = match plane_roles.has_specialization() {
            true => SpecializerKind::LoadOverlapCompute,
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

    fn must_check_if_loader(&self) -> bool {
        match self.kind {
            SpecializerKind::LoadOverlapCompute => self.plane_roles.compute_only > 0,
            SpecializerKind::NoSpecialization => false,
        }
    }

    fn must_check_if_computer(&self) -> bool {
        match self.kind {
            SpecializerKind::LoadOverlapCompute => self.plane_roles.load_only > 0,
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

    pub fn must_check_if_loader(&self) -> comptime_type!(bool) {
        comptime!(self.specializer_config.must_check_if_loader())
    }

    pub fn is_loader(&self, plane_id: u32) -> bool {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOverlapCompute => {
                let plane_roles = self.specializer_config.plane_roles;
                plane_id < comptime!(plane_roles.load_only + plane_roles.overlap)
            }
            SpecializerKind::NoSpecialization => {
                comptime!(unreachable!("Should call must_check_if_loader prior"))
            }
        }
    }

    pub fn must_check_if_computer(&self) -> comptime_type!(bool) {
        comptime!(self.specializer_config.must_check_if_computer())
    }

    pub fn is_computer(&self, plane_id: u32) -> bool {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOverlapCompute => {
                plane_id >= self.specializer_config.plane_roles.load_only
            }
            SpecializerKind::NoSpecialization => {
                comptime!(unreachable!("Should call must_check_if_computer prior"))
            }
        }
    }

    pub fn plane_id_to_loader_index(&self, plane_id: u32) -> u32 {
        plane_id
    }

    pub fn plane_id_to_computer_index(&self, plane_id: u32) -> u32 {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOverlapCompute => {
                plane_id - self.specializer_config.plane_roles.load_only
            }
            SpecializerKind::NoSpecialization => plane_id,
        }
    }

    pub fn loader_index_to_plane_id(&self, loader_index: u32) -> u32 {
        loader_index
    }

    pub fn computer_index_to_plane_id(&self, computer_index: u32) -> u32 {
        match comptime!(self.specializer_config.kind) {
            SpecializerKind::LoadOverlapCompute => {
                self.specializer_config.plane_roles.load_only + computer_index
            }
            SpecializerKind::NoSpecialization => computer_index,
        }
    }
}

////////////////////////////

// #[cube]
// pub trait SpecializationRule: CubeType {
//     fn must_check_if_loader(this: &Self) -> comptime_type!(bool);

//     fn is_loader(this: &Self, plane_id: u32) -> bool;

//     fn must_check_if_computer(this: &Self) -> comptime_type!(bool);

//     fn is_computer(this: &Self, plane_id: u32) -> bool;

//     fn plane_id_to_loader_index(_this: &Self, plane_id: u32) -> u32;

//     fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32;

//     fn loader_index_to_plane_id(_this: &Self, loader_index: u32) -> u32;

//     fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32;
// }

// #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
// /// First plane ids are given to load only, subsequent to overlap, and last to compute only
// pub struct ThresholdSpecializer {
//     #[cube(comptime)]
//     plane_roles: PlaneRoles,
// }

// #[cube]
// impl SpecializationRule for ThresholdSpecializer {
//     fn must_check_if_loader(this: &Self) -> comptime_type!(bool) {
//         comptime!(this.plane_roles.compute_only > 0)
//     }

//     fn is_loader(this: &Self, plane_id: u32) -> bool {
//         plane_id < comptime!(this.plane_roles.load_only + this.plane_roles.overlap)
//     }

//     fn must_check_if_computer(this: &Self) -> comptime_type!(bool) {
//         comptime!(this.plane_roles.load_only > 0)
//     }

//     fn is_computer(this: &Self, plane_id: u32) -> bool {
//         plane_id >= this.plane_roles.load_only
//     }

//     fn plane_id_to_loader_index(_this: &Self, plane_id: u32) -> u32 {
//         plane_id
//     }

//     fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32 {
//         plane_id - this.plane_roles.load_only
//     }

//     fn loader_index_to_plane_id(_this: &Self, loader_index: u32) -> u32 {
//         loader_index
//     }

//     fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32 {
//         this.plane_roles.load_only + computer_index
//     }
// }

// #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
// pub struct NoSpecializer {}

// #[cube]
// impl SpecializationRule for NoSpecializer {
//     fn must_check_if_loader(_this: &Self) -> comptime_type!(bool) {
//         false
//     }

//     fn is_loader(_this: &Self, _plane_id: u32) -> bool {
//         true.runtime()
//     }

//     fn must_check_if_computer(_this: &Self) -> comptime_type!(bool) {
//         false
//     }

//     fn is_computer(_this: &Self, _plane_id: u32) -> bool {
//         true.runtime()
//     }

//     fn plane_id_to_loader_index(_this: &Self, plane_id: u32) -> u32 {
//         plane_id
//     }

//     fn plane_id_to_computer_index(_this: &Self, plane_id: u32) -> u32 {
//         plane_id
//     }

//     fn loader_index_to_plane_id(_this: &Self, loader_index: u32) -> u32 {
//         loader_index
//     }

//     fn computer_index_to_plane_id(_this: &Self, computer_index: u32) -> u32 {
//         computer_index
//     }
// }

// #[derive(CubeType)]
// pub enum Specializer {
//     NoSpecializer(NoSpecializer),
//     ThresholdSpecializer(ThresholdSpecializer),
// }

// #[cube]
// impl SpecializationRule for Specializer {
//     fn must_check_if_loader(this: &Self) -> comptime_type!(bool) {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::must_check_if_loader(ns)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::must_check_if_loader(ts)
//             }
//         }
//     }

//     fn is_loader(this: &Self, plane_id: u32) -> bool {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::is_loader(ns, plane_id)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::is_loader(ts, plane_id)
//             }
//         }
//     }

//     fn must_check_if_computer(this: &Self) -> comptime_type!(bool) {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::must_check_if_computer(ns)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::must_check_if_computer(ts)
//             }
//         }
//     }

//     fn is_computer(this: &Self, plane_id: u32) -> bool {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::is_computer(ns, plane_id)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::is_computer(ts, plane_id)
//             }
//         }
//     }

//     fn plane_id_to_loader_index(this: &Self, plane_id: u32) -> u32 {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::plane_id_to_loader_index(ns, plane_id)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::plane_id_to_loader_index(ts, plane_id)
//             }
//         }
//     }

//     fn plane_id_to_computer_index(this: &Self, plane_id: u32) -> u32 {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::plane_id_to_computer_index(ns, plane_id)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::plane_id_to_computer_index(
//                     ts, plane_id,
//                 )
//             }
//         }
//     }

//     fn loader_index_to_plane_id(this: &Self, loader_index: u32) -> u32 {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::loader_index_to_plane_id(ns, loader_index)
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::loader_index_to_plane_id(
//                     ts,
//                     loader_index,
//                 )
//             }
//         }
//     }

//     fn computer_index_to_plane_id(this: &Self, computer_index: u32) -> u32 {
//         match this {
//             Specializer::NoSpecializer(ns) => {
//                 <NoSpecializer as SpecializationRule>::computer_index_to_plane_id(
//                     ns,
//                     computer_index,
//                 )
//             }
//             Specializer::ThresholdSpecializer(ts) => {
//                 <ThresholdSpecializer as SpecializationRule>::computer_index_to_plane_id(
//                     ts,
//                     computer_index,
//                 )
//             }
//         }
//     }
// }

// #[cube]
// impl Specializer {
//     pub fn new(#[comptime] plane_roles: PlaneRoles) -> Self {
//         match plane_roles.has_specialization() {
//             true => Specializer::new_ThresholdSpecializer(ThresholdSpecializer { plane_roles }),
//             false => Specializer::new_NoSpecializer(NoSpecializer {}),
//         }
//     }

//     pub fn must_check_if_loader(&self) -> comptime_type!(bool) {
//         <Specializer as SpecializationRule>::must_check_if_loader(&self)
//     }

//     pub fn must_check_if_computer(&self) -> comptime_type!(bool) {
//         <Specializer as SpecializationRule>::must_check_if_computer(&self)
//     }

//     pub fn is_loader(&self, plane_id: u32) -> bool {
//         <Specializer as SpecializationRule>::is_loader(&self, plane_id)
//     }

//     pub fn is_computer(&self, plane_id: u32) -> bool {
//         <Specializer as SpecializationRule>::is_computer(&self, plane_id)
//     }

//     pub fn plane_id_to_loader_index(&self, plane_id: u32) -> u32 {
//         <Specializer as SpecializationRule>::plane_id_to_loader_index(&self, plane_id)
//     }

//     pub fn plane_id_to_computer_index(&self, plane_id: u32) -> u32 {
//         <Specializer as SpecializationRule>::plane_id_to_computer_index(&self, plane_id)
//     }

//     pub fn loader_index_to_plane_id(&self, loader_index: u32) -> u32 {
//         <Specializer as SpecializationRule>::loader_index_to_plane_id(&self, loader_index)
//     }

//     pub fn computer_index_to_plane_id(&self, computer_index: u32) -> u32 {
//         <Specializer as SpecializationRule>::computer_index_to_plane_id(&self, computer_index)
//     }
// }
