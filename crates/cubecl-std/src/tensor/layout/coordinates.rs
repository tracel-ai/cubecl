use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

/// A set of coordinates used in layouts. Contains some utilities for comptime inspection.
#[cube]
pub trait Coordinates: CubeType + Clone {
    /// Add two coordinates together and return the result.
    fn add(this: Self, other: Self) -> Self;
    /// Check whether `pos` is fully contained within `bounds`.
    fn is_in_bounds(pos: Self, bounds: Self) -> bool;
    /// The origin (zero) coordinates. `this` may be used as a reference coordinate for dynamically
    /// sized layouts.
    fn origin(this: &Self) -> Self;
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
    fn add(this: Self, other: Self) -> Self {
        this + other
    }
    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        pos < bounds
    }
    fn origin(_this: &Self) -> Self {
        0u32.runtime()
    }
}
#[cube]
impl Coordinates for Coords2d {
    fn add(this: Self, other: Self) -> Self {
        (this.0 + other.0, this.1 + other.1)
    }
    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        pos.0 < bounds.0 && pos.1 < bounds.1
    }
    fn origin(_this: &Self) -> Self {
        (0u32, 0u32).runtime()
    }
}
#[cube]
impl Coordinates for Coords3d {
    fn add(this: Self, other: Self) -> Self {
        (this.0 + other.0, this.1 + other.1, this.2 + other.2)
    }
    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        pos.0 < bounds.0 && pos.1 < bounds.1 && pos.2 < bounds.2
    }
    fn origin(_this: &Self) -> Self {
        (0u32, 0u32, 0u32).runtime()
    }
}
#[cube]
impl Coordinates for Coords4d {
    fn add(this: Self, other: Self) -> Self {
        (
            this.0 + other.0,
            this.1 + other.1,
            this.2 + other.2,
            this.3 + other.3,
        )
    }
    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        pos.0 < bounds.0 && pos.1 < bounds.1 && pos.2 < bounds.2 && pos.3 < bounds.3
    }
    fn origin(_this: &Self) -> Self {
        (0u32, 0u32, 0u32, 0u32).runtime()
    }
}
#[cube]
impl Coordinates for Coords5d {
    fn add(this: Self, other: Self) -> Self {
        (
            this.0 + other.0,
            this.1 + other.1,
            this.2 + other.2,
            this.3 + other.3,
            this.4 + other.4,
        )
    }
    fn is_in_bounds(pos: Self, bound: Self) -> bool {
        pos.0 < bound.0 && pos.1 < bound.1 && pos.2 < bound.2 && pos.3 < bound.3 && pos.4 < bound.4
    }
    fn origin(_this: &Self) -> Self {
        (0u32, 0u32, 0u32, 0u32, 0u32).runtime()
    }
}
#[cube]
impl Coordinates for CoordsDyn {
    fn add(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(*this.index(i) + *other.index(i));
        }

        out
    }

    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        let rank = comptime![pos.len()];
        let mut out = true;

        #[unroll]
        for i in 0..rank {
            out &= *pos.index(i) < *bounds.index(i);
        }

        out
    }

    fn origin(this: &Self) -> Self {
        let rank = comptime![this.len()];
        let mut origin = Sequence::new();

        #[unroll]
        for _ in 0..rank {
            origin.push(0u32.runtime())
        }

        origin
    }
}
