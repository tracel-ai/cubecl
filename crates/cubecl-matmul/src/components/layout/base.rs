use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use super::Coordinates;

#[cube]
pub trait Layout: CubeType + Clone + Send + Sync + 'static {
    type Coordinates: Coordinates;

    fn to_linear(this: &Self, coords: Self::Coordinates) -> u32;
    fn to_linear_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool);
    fn from_linear(this: &Self, idx: u32) -> Self::Coordinates;
    fn shape(this: &Self) -> Self::Coordinates;
}

#[cube]
pub trait LayoutTransform<From: Coordinates, To: Coordinates>:
    CubeType + Clone + Send + Sync + 'static
{
    fn transform_to(this: &Self, coords: From) -> To;
    fn transform_from(this: &Self, coords: To) -> From;
}

#[derive(Clone, Copy)]
pub struct NoTransform;

impl CubeType for NoTransform {
    type ExpandType = Self;
}

impl IntoMut for NoTransform {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for NoTransform {}

#[cube]
impl<T: Coordinates> LayoutTransform<T, T> for NoTransform {
    fn transform_to(_this: &Self, coords: T) -> T {
        coords
    }

    fn transform_from(_this: &Self, coords: T) -> T {
        coords
    }
}
