use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulIdent,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod,
    tensor::{
        layout::{Coords3d, Layout},
        r#virtual::VirtualTensor,
    },
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig,
        global::layout::{unwrap, virtual_layout},
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D weight tensor of shape `(out_c, (k_h, k_w, in_c))` to a col-major 2D matmul tile with
/// shape `(n, k)`
#[derive(CubeType, Clone)]
pub struct WeightGlobalLayout {
    /// Stride of `out_c`
    pub stride_out_c: u32,
    /// Stride of `k_h`, `k_w`
    pub strides_spatial: Sequence<u32>,
    /// Stride of `in_c`
    pub stride_in_c: u32,

    /// Number of channels, including padding, used for decomposing `k`
    pub channels: FastDivmod,

    /// Shape of the conceptual `k` size, including padding
    pub shape_k: u32,
    /// Shape of the conceptual `n` size, or `out_c`
    pub shape_n: u32,

    /// Size of the convolution kernel
    #[cube(comptime)]
    pub kernel_size: [u32; 3],
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl WeightGlobalLayout {
    pub fn new<E: Numeric, G: GlobalConfig>(
        tensor: &VirtualTensor<E>,
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> WeightGlobalLayout {
        let spatial_dims = comptime![config.dimensionality().num_dims()];
        let mut strides_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
        }

        let stride_out_c = tensor.stride(0);
        let stride_in_c = tensor.stride(spatial_dims + 1);

        WeightGlobalLayout {
            stride_out_c,
            strides_spatial,
            stride_in_c,
            shape_k: args.shape_k,
            shape_n: args.shape_n,
            channels: args.padded_channels,
            kernel_size: config.kernel_size,
            config: config.global_memory_config(MatmulIdent::Rhs),
        }
    }
}

#[cube]
impl Layout for WeightGlobalLayout {
    type Coordinates = Coords3d;

    fn to_linear_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        let (_, k, n) = coords;

        let (mut rem, in_c) = this.channels.div_mod(k);

        let spatial_dims = comptime![this.strides_spatial.len()];
        let mut kernel_pos = Sequence::<u32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![this.kernel_size[dim as usize]];
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

        let line_size = comptime![this.config.global_line_size];
        read_pos / line_size
    }

    fn to_linear_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let linear_pos = Self::to_linear_pos(this, coords);

        let (_, k, n) = coords;
        let check_k = comptime![this.config.check_row_bounds];
        let check_n = comptime![this.config.check_col_bounds];
        let in_bounds = (!check_k || k < this.shape_k) && (!check_n || n < this.shape_n);

        (linear_pos, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (1, this.shape_k, this.shape_n)
    }
}

virtual_layout!(WeightGlobalLayout, WeightGlobalLayoutExpand);
