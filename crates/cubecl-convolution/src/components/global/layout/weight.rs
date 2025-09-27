use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulIdent,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod,
    tensor::{
        layout::{Coords2d, Layout, LayoutExpand},
        r#virtual::VirtualTensor,
    },
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig,
        global::layout::{NhwcCoords, unwrap},
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D weight tensor of shape `(out_c, (k_h, k_w, in_c))` to a col-major 2D matmul tile with
/// shape `(n, k)`
#[derive(CubeType, Clone)]
pub struct WeightLayout {
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
impl WeightLayout {
    pub fn new<E: Numeric, G: GlobalConfig>(
        tensor: &VirtualTensor<E>,
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> WeightLayout {
        let spatial_dims = comptime![config.dimensionality().num_dims()];
        let mut strides_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
        }

        let stride_out_c = tensor.stride(0);
        let stride_in_c = tensor.stride(spatial_dims + 1);

        WeightLayout {
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
impl Layout for WeightLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> NhwcCoords {
        let (k, n) = coords;

        let (mut rem, in_c) = self.channels.div_mod(k);

        let spatial_dims = comptime![self.strides_spatial.len()];
        let mut kernel_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![self.kernel_size[dim as usize]];
            let k_pos = rem % ksize;
            rem /= ksize;

            kernel_pos.push(k_pos as i32);
        }

        let kernel_pos = kernel_pos.rev();

        NhwcCoords {
            batch: n,
            spatial: kernel_pos,
            channel: in_c,
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.shape_k, self.shape_n)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (k, n) = pos;
        let check_k = comptime![self.config.check_row_bounds];
        let check_n = comptime![self.config.check_col_bounds];
        (!check_k || k < self.shape_k) && (!check_n || n < self.shape_n)
    }
}
