use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::*;

/// Slice the layout at a specific batch, and reduce its dimensionality
/// Not general enough to be in cubecl-std
#[derive(CubeType, Clone, Copy)]
pub struct SliceIndex {
    offset: u32,
    shape: Coords2d,
}

#[cube]
impl SliceIndex {
    pub fn new(offset: u32, shape: Coords3d) -> Self {
        let (_, rows, cols) = shape;
        SliceIndex {
            offset,
            shape: (rows, cols),
        }
    }
}

#[cube]
impl Layout for SliceIndex {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        (self.offset, row, col)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // we don't check batch
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }
}
