use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait PlaneMapper {
    fn plane_id() -> u32;
    fn plane_unit() -> u32;
    fn num_planes() -> u32;
    fn plane_dim() -> u32;
}
