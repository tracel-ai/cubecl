use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::global::memory::GlobalMemoryConfig;
use cubecl_std::{
    FastDivmod,
    tensor::layout::{Coords2d, Layout, LayoutExpand},
};

use crate::{
    components::global::{
        layout::{NhwcCoords, cast_seq},
        read::im2col_tma::div_mod_seq,
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D NHWC out tensor of shape `((n, h, w), c)` to a col-major 2D matmul tile with
/// shape `(m, n)`
#[derive(CubeType, Clone)]
pub struct OutLayout {
    /// Shape of DHW
    pub shape_out: Sequence<FastDivmod>,

    /// Shape of the conceptual `m` size
    pub shape_m: u32,
    /// Shape of the conceptual `n`size, or channels
    pub shape_n: u32,

    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl OutLayout {
    pub fn new(args: &RuntimeArgs, #[comptime] config: GlobalMemoryConfig) -> OutLayout {
        OutLayout {
            shape_out: args.shape_out.clone(),
            shape_m: args.shape_m,
            shape_n: args.shape_n,
            config,
        }
    }
}

#[cube]
impl Layout for OutLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> NhwcCoords {
        let (view_m, view_n) = coords;
        let (batch, spatial) = div_mod_seq(view_m, &self.shape_out);

        NhwcCoords {
            batch,
            spatial: cast_seq(spatial),
            channel: view_n,
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.shape_m, self.shape_n)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (m, n) = pos;
        let check_m = comptime![self.config.check_row_bounds];
        let check_n = comptime![self.config.check_col_bounds];
        (!check_m || m < self.shape_m) && (!check_n || n < self.shape_n)
    }
}
