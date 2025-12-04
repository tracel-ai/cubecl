use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{
    FastDivmod,
    tensor::layout::{Coords3d, Layout, LayoutExpand},
};

use crate::components::global::layout::NhwcCoords;

#[derive(CubeType, CubeLaunch)]
pub struct TmaWeightLayout {
    padded_channels: FastDivmod,
    #[cube(comptime)]
    kernel_size: Vec<u32>,
}

#[cube]
impl TmaWeightLayout {
    pub fn new(padded_channels: FastDivmod, #[comptime] kernel_size: Vec<u32>) -> Self {
        TmaWeightLayout {
            padded_channels,
            kernel_size,
        }
    }
}

#[cube]
impl Layout for TmaWeightLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, k, n) = pos;
        let (mut k_idx, in_c) = self.padded_channels.div_mod(k);
        let k_rank = comptime![self.kernel_size.len() as u32];
        let mut k_pos = Sequence::new();

        #[unroll]
        for i in 0..k_rank {
            let dim = comptime![k_rank - i - 1];
            let k_size = comptime![self.kernel_size[dim as usize]];
            k_pos.push((k_idx % k_size) as i32);
            k_idx /= k_size;
        }

        NhwcCoords {
            batch: n,
            spatial: k_pos.rev(),
            channel: in_c,
        }
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
