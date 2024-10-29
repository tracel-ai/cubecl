use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
/// Gives indexes for the current unit
///
/// Typically corresponds directly to unit positions within the cube, but
/// can be customized
pub trait PlaneMapper {
    /// In which plane the unit lies
    ///
    /// Typically UNIT_POS_Y
    fn plane_id() -> u32;
    /// The position of the unit within the plane
    ///
    /// Typically UNIT_POS_X
    fn plane_unit() -> u32;
}

/// A config for a matmul
///
/// Useful to aggregate many trait bounds
pub trait MatmulConfig:
    CubeType + Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug + IntoRuntime
{
}
