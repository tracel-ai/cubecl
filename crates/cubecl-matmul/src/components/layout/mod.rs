use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

#[cube]
pub trait Coordinates: CubeType {
    fn rank(this: Self) -> comptime_type!(u32);
    fn dim(this: Self, #[comptime] dim: u32) -> u32;
}

#[cube]
pub trait Layout: CubeType {
    type Coordinates: Coordinates;

    fn to_linear(&self, coords: Self::Coordinates) -> u32;
    #[allow(clippy::wrong_self_convention)]
    fn from_linear(&self, idx: u32) -> Self::Coordinates;
}

type Coords2d = (u32, u32);

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

#[derive(CubeType)]
pub struct Swizzle<Inner: Layout<Coordinates = Coords2d>> {
    #[cube(comptime)]
    pub mode: TensorMapSwizzle,
    pub inner: Inner,
}

#[cube]
impl<Inner: Layout<Coordinates = Coords2d>> Layout for Swizzle<Inner> {
    type Coordinates = Coords2d;

    fn to_linear(&self, coords: Self::Coordinates) -> u32 {
        let coords = swizzle(coords);
        self.inner.to_linear(coords)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_linear(&self, idx: u32) -> Self::Coordinates {
        let coords = self.inner.from_linear(idx);
        unswizzle(coords)
    }
}
