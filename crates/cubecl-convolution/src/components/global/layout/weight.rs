use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulIdent,
    layout::{Coords2d, Layout},
};
use cubecl_std::{FastDivmod, tensor::r#virtual::VirtualTensor};

use crate::components::{
    ConvGemmConfig,
    global::layout::{unwrap, virtual_layout},
};

#[derive(CubeType, Clone)]
pub struct WeightGlobalLayout<C: ConvGemmConfig> {
    pub stride_out_c: u32,
    pub strides_spatial: Sequence<u32>,
    pub stride_in_c: u32,

    pub channels: FastDivmod,

    pub shape_k: u32,
    pub shape_n: u32,

    #[cube(comptime)]
    pub config: C,
}

#[cube]
impl<C: ConvGemmConfig> WeightGlobalLayout<C> {
    pub fn new<E: Numeric>(
        tensor: &VirtualTensor<E>,
        shape_k: u32,
        shape_n: u32,
        channels: FastDivmod,
        #[comptime] config: C,
    ) -> WeightGlobalLayout<C> {
        let spatial_dims = comptime![config.dimensionality().num_dims()];
        let mut strides_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
        }

        let stride_out_c = tensor.stride(0);
        let stride_in_c = tensor.stride(spatial_dims + 1);

        WeightGlobalLayout::<C> {
            stride_out_c,
            strides_spatial,
            stride_in_c,
            shape_k,
            shape_n,
            channels,
            config,
        }
    }
}

#[cube]
impl<C: ConvGemmConfig> Layout for WeightGlobalLayout<C> {
    type Coordinates = Coords2d;

    fn to_linear_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        let (k, n) = coords;

        let (mut rem, in_c) = this.channels.div_mod(k);

        let spatial_dims = comptime![this.strides_spatial.len()];
        let mut kernel_pos = Sequence::<u32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![this.config.kernel_size(dim)];
            let k_pos = rem % ksize;
            rem /= ksize;

            kernel_pos.push(k_pos);
        }

        let kernel_pos = kernel_pos.rev();

        let mut read_pos = n * this.stride_out_c + in_c * this.stride_in_c;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            read_pos += *kernel_pos.index(i) * *this.strides_spatial.index(i);
        }

        read_pos
    }

    fn to_linear_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let linear_pos = Self::to_linear_pos(this, coords);

        let (k, n) = coords;
        let check_k = comptime![this.config.check_row_bounds(MatmulIdent::Rhs)];
        let check_n = comptime![this.config.check_col_bounds(MatmulIdent::Rhs)];
        let in_bounds = (!check_k || k < this.shape_k) && (!check_n || n < this.shape_n);

        (linear_pos, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.shape_k, this.shape_n)
    }
}

virtual_layout!(WeightGlobalLayout, WeightGlobalLayoutExpand);
