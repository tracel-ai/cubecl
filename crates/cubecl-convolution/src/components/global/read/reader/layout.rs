use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{
    FastDivmod,
    tensor::layout::{Coords2d, Coords3d, Layout, LayoutExpand},
};

#[derive(CubeType)]
pub struct TmaWeightLayout {
    padded_channels: FastDivmod,
}

#[cube]
impl TmaWeightLayout {
    pub fn new(padded_channels: FastDivmod) -> Self {
        TmaWeightLayout { padded_channels }
    }
}

#[cube]
impl Layout for TmaWeightLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (k, n) = pos;
        let (k_idx, in_c) = self.padded_channels.div_mod(k);
        (n, k_idx, in_c)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
