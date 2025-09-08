use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::global::memory::GlobalMemoryConfig;
use cubecl_std::{
    FastDivmod,
    tensor::{
        layout::{Coords3d, Layout},
        r#virtual::VirtualTensor,
    },
};

use crate::components::global::{
    layout::{unwrap, virtual_layout},
    load::im2col_tma::div_mod_seq,
};

/// Maps a 4D NHWC out tensor of shape `((n, h, w), c)` to a col-major 2D matmul tile with
/// shape `(m, n)`
#[derive(CubeType, Clone)]
pub struct NhwcOutGlobalLayout {
    /// Stride of N
    pub stride_n: u32,
    /// Strides of DHW
    pub strides_spatial: Sequence<u32>,
    /// Stride of C
    pub stride_c: u32,

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
impl NhwcOutGlobalLayout {
    pub fn new<E: Numeric>(
        tensor: &VirtualTensor<E, ReadWrite>,
        shape_m: u32,
        shape_n: u32,
        shape_out: Sequence<FastDivmod>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> NhwcOutGlobalLayout {
        let spatial_dims = comptime![shape_out.len()];
        let mut strides_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
        }

        let stride_n = tensor.stride(0);
        let stride_c = tensor.stride(spatial_dims + 1);

        NhwcOutGlobalLayout {
            stride_n,
            strides_spatial,
            stride_c,
            shape_out,
            shape_m,
            shape_n,
            config,
        }
    }
}

#[cube]
impl Layout for NhwcOutGlobalLayout {
    type Coordinates = Coords3d;

    fn to_linear_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        let (_, view_m, view_n) = coords;

        let (n, out_pos) = div_mod_seq(view_m, &this.shape_out);

        let spatial_dims = comptime![this.shape_out.len()];
        let c = view_n;

        let mut write_pos = n * this.stride_n + c * this.stride_c;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            write_pos += *out_pos.index(i) as u32 * *this.strides_spatial.index(i);
        }

        write_pos / this.config.global_line_size
    }

    fn to_linear_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let linear_pos = Self::to_linear_pos(this, coords);

        let (_, m, n) = coords;
        let check_m = comptime![this.config.check_row_bounds];
        let check_n = comptime![this.config.check_col_bounds];
        let in_bounds = (!check_m || m < this.shape_m) && (!check_n || n < this.shape_n);

        (linear_pos, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (1, this.shape_m, this.shape_n)
    }
}

virtual_layout!(NhwcOutGlobalLayout, NhwcOutGlobalLayoutExpand);
