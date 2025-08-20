use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_matmul::components::{
    MatmulIdent,
    layout::{Coords2d, Layout},
};
use cubecl_std::{FastDivmod, tensor::r#virtual::VirtualTensor};

use crate::components::{ConvGemmConfig, global::load::im2col_tma::div_mod_seq};

#[derive(CubeType, Clone)]
pub struct Im2colGlobalLayout<C: ConvGemmConfig> {
    pub stride_batch: u32,
    pub strides_spatial: Sequence<u32>,
    pub stride_channel: u32,

    pub shapes_spatial: Sequence<u32>,
    pub shape_channel: u32,

    pub shape_out: Sequence<FastDivmod>,

    pub shape_m: u32,
    pub shape_k: u32,

    #[cube(comptime)]
    pub config: C,
}

#[cube]
impl<C: ConvGemmConfig> Im2colGlobalLayout<C> {
    pub fn new<E: Numeric>(
        tensor: &VirtualTensor<E>,
        shape_out: Sequence<FastDivmod>,
        shape_m: u32,
        shape_k: u32,
        #[comptime] config: C,
    ) -> Im2colGlobalLayout<C> {
        let spatial_dims = comptime![shape_out.len()];
        let mut strides_spatial = Sequence::new();
        let mut shapes_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
            shapes_spatial.push(tensor.shape(i + 1));
        }

        let stride_batch = tensor.stride(0);
        let stride_channel = tensor.stride(spatial_dims + 1);

        let shape_channel = tensor.shape(spatial_dims + 1);

        Im2colGlobalLayout::<C> {
            stride_batch,
            strides_spatial,
            stride_channel,
            shapes_spatial,
            shape_channel,
            shape_out,
            shape_m,
            shape_k,
            config,
        }
    }
}

#[cube]
impl<C: ConvGemmConfig> Layout for Im2colGlobalLayout<C> {
    type Coordinates = Coords2d;

    fn to_linear(this: &Self, coords: Self::Coordinates) -> u32 {
        Self::to_linear_checked(this, coords).0
    }

    fn from_linear(this: &Self, idx: u32) -> Self::Coordinates {
        let k = idx % this.shape_k;
        let m = idx / this.shape_k;
        (m, k)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.shape_m, this.shape_k)
    }

    fn to_linear_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let (view_m, view_k) = coords;

        let (batch, out_offs) = div_mod_seq(view_m, &this.shape_out);

        let channel = view_k % this.shape_channel;
        let mut rem = view_k / this.shape_channel;

        let spatial_dims = comptime![this.shapes_spatial.len()];
        let mut in_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![this.config.kernel_size(dim)];
            let k_pos = rem % ksize;
            rem /= ksize;

            let out_pos = *out_offs.index(dim);
            let stride = comptime![this.config.stride(dim)];
            let dilate = comptime![this.config.dilation(dim)];
            let pad = comptime![this.config.padding(dim)];

            let pos = (out_pos * stride + k_pos * dilate) as i32 - pad;
            in_pos.push(pos);
        }

        let in_pos = in_pos.rev();

        let has_padding = comptime! {
            let mut has_padding = false;
            for i in 0..spatial_dims {
                has_padding |= this.config.padding(i) != 0;
            }
            has_padding
        };

        let m_in_bounds =
            comptime!(!this.config.check_row_bounds(MatmulIdent::Lhs)) || view_m < this.shape_m;
        let k_in_bounds =
            comptime!(!this.config.check_col_bounds(MatmulIdent::Lhs)) || view_k < this.shape_k;
        let mut spatial_in_bounds = true;

        if has_padding {
            #[unroll]
            for i in 0..spatial_dims {
                let i = unwrap(i);
                let pos = *in_pos.index(i);
                spatial_in_bounds &= pos >= 0 && (pos as u32) < *this.shapes_spatial.index(i);
            }
        }

        let in_bounds = m_in_bounds && k_in_bounds && spatial_in_bounds;

        let mut read_pos = batch * this.stride_batch + channel * this.stride_channel;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            read_pos += *in_pos.index(i) as u32 * *this.strides_spatial.index(i);
        }

        (read_pos, in_bounds)
    }
}

#[allow(unused_variables)]
#[cube]
fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}
