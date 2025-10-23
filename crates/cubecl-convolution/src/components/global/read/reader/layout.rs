use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{
    FastDivmod,
    tensor::layout::{Coords3d, Layout, LayoutExpand},
};

#[derive(CubeType, CubeLaunch)]
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
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, k, n) = pos;
        let (k_idx, in_c) = self.padded_channels.div_mod(k);
        (n, k_idx, in_c)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Dummy layout for launching, to be exited out later with `as_tensor_map`.
#[derive(CubeType, CubeLaunch)]
pub struct TmaDummyLayout {}

#[cube]
impl Layout for TmaDummyLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        pos
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
