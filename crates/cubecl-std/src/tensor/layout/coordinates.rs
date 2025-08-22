use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

/// A set of coordinates used in layouts. Contains some utilities for comptime inspection.
#[cube]
pub trait Coordinates: CubeType + Clone {
    fn rank(this: Self) -> comptime_type!(u32);
    fn dim(this: Self, #[comptime] dim: u32) -> u32;
}

// Aliases for convenience and semantic clarity
pub type Coords1d = u32;
pub type Coords2d = (u32, u32);
pub type Coords3d = (u32, u32, u32);
pub type Coords4d = (u32, u32, u32, u32);
pub type Coords5d = (u32, u32, u32, u32, u32);
pub type CoordsDyn = Sequence<u32>;

#[cube]
impl Coordinates for Coords1d {
    fn rank(_this: Self) -> comptime_type!(u32) {
        1
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        match dim {
            0 => this,
            _ => panic!("Invalid dim"),
        }
    }
}

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

#[cube]
impl Coordinates for Coords3d {
    fn rank(_this: Self) -> comptime_type!(u32) {
        3
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        match dim {
            0 => this.0,
            1 => this.1,
            2 => this.2,
            _ => panic!("Invalid dim"),
        }
    }
}

#[cube]
impl Coordinates for Coords4d {
    fn rank(_this: Self) -> comptime_type!(u32) {
        4
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        match dim {
            0 => this.0,
            1 => this.1,
            2 => this.2,
            3 => this.3,
            _ => panic!("Invalid dim"),
        }
    }
}

#[cube]
impl Coordinates for Coords5d {
    fn rank(_this: Self) -> comptime_type!(u32) {
        5
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        match dim {
            0 => this.0,
            1 => this.1,
            2 => this.2,
            3 => this.3,
            4 => this.4,
            _ => panic!("Invalid dim"),
        }
    }
}

#[cube]
impl Coordinates for CoordsDyn {
    fn rank(this: Self) -> comptime_type!(u32) {
        this.len()
    }

    fn dim(this: Self, #[comptime] dim: u32) -> u32 {
        *this.index(dim)
    }
}
