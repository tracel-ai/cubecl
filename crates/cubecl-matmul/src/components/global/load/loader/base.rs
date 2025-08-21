use cubecl_core::prelude::CubeType;

use crate::components::layout::Coordinates;

pub trait Loader: CubeType {
    type Coordinates: Coordinates;
}
