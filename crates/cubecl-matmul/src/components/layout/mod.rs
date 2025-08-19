use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
pub trait Coordinates: CubeType {
    fn rank(this: Self) -> comptime_type!(u32);
    fn dim(this: Self, #[comptime] dim: u32) -> u32;
}

#[cube]
pub trait Layout: CubeType + Clone + Send + Sync + 'static {
    type Coordinates: Coordinates;

    fn to_linear(this: &Self, coords: Self::Coordinates) -> u32;
    fn to_linear_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool);
    fn from_linear(this: &Self, idx: u32) -> Self::Coordinates;
    fn shape(this: &Self) -> Self::Coordinates;
}

pub type Coords2d = (u32, u32);

#[cube]
impl Coordinates for Coords2d {
    fn rank(_this: Self) -> comptime_type!(u32) {
        2
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        match dim {
            0 => this.0,
            1 => this.1,
            _ => panic!("Invalid dim"),
        }
    }
}
