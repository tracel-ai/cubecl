use cubecl_core::prelude::CubeType;
use cubecl_std::tensor::layout::Coordinates;

pub trait Loader: CubeType {
    type Coordinates: Coordinates;
}
