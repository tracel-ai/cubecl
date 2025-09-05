use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulIdent,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod,
    tensor::layout::{Coords3d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand},
};

use crate::{
    components::{
        ConvolutionConfig,
        global::{
            layout::{NhwcCoords, unwrap, virtual_layout},
            load::im2col_tma::div_mod_seq,
        },
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D NHWC tensor to a 2D column matrix using the im2col transformation
/// It first decomposes the `(m, k)` matrix into `((n, out_h, out_w), (k_h, k_w, c))`, then applies
/// the convolution parameters to calculate the position in the input tensor for that kernel element.
#[derive(CubeType, Clone)]
pub struct Im2colGlobalLayout {
    /// Shape of output DHW
    pub shape_out: Sequence<FastDivmod>,
    /// Shape of channel, for decomposing k
    pub shape_channel: FastDivmod,

    /// Shape of the combined `m` dimension, including padding
    pub shape_m: u32,
    /// Shape of the combined `k` dimension, including padding
    pub shape_k: u32,

    /// Size of the convolution kernel in DHW
    #[cube(comptime)]
    pub kernel_size: [u32; 3],
    /// Stride of the convolution in DHW
    #[cube(comptime)]
    pub stride: [u32; 3],
    /// Dilation applied to the kernel positions in DHW
    #[cube(comptime)]
    pub dilation: [u32; 3],
    /// Padding applied to the convolution in DHW
    /// The input position will be offset from the output by `-padding`
    #[cube(comptime)]
    pub padding: [i32; 3],
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl Im2colGlobalLayout {
    pub fn new<G: GlobalConfig>(
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> Im2colGlobalLayout {
        let shape_out = args.shape_out.clone();

        Im2colGlobalLayout {
            shape_out,
            shape_channel: args.shape_channel,
            shape_m: args.shape_m,
            shape_k: args.shape_k,
            kernel_size: config.kernel_size,
            stride: config.stride,
            dilation: config.dilation,
            padding: config.padding,
            config: config.global_memory_config(MatmulIdent::Lhs),
        }
    }
}

#[cube]
impl Layout for Im2colGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> NhwcCoords {
        let (_, view_m, view_k) = pos;

        let (batch, out_offs) = div_mod_seq(view_m, &this.shape_out);

        let (mut rem, channel) = this.shape_channel.div_mod(view_k);

        let spatial_dims = comptime![this.shape_out.len()];
        let mut in_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![this.kernel_size[dim as usize]];
            let k_pos = rem % ksize;
            rem /= ksize;

            let out_pos = *out_offs.index(dim);
            let stride = comptime![this.stride[dim as usize]];
            let dilate = comptime![this.dilation[dim as usize]];
            let pad = comptime![this.padding[dim as usize]];

            let pos = (out_pos * stride + k_pos * dilate) as i32 - pad;
            in_pos.push(pos);
        }

        let in_pos = in_pos.rev();

        NhwcCoords {
            batch,
            spatial: in_pos,
            channel,
        }
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (1, this.shape_m, this.shape_k)
    }

    fn to_source_pos_checked(this: &Self, pos: Self::Coordinates) -> (NhwcCoords, bool) {
        (this.to_source_pos(pos), this.is_in_bounds(pos))
    }

    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool {
        let (_, view_m, view_k) = pos;
        // Shouldn't be relied on because it doesn't check spatial
        let m_in_bounds = comptime!(!this.config.check_row_bounds) || view_m < this.shape_m;
        let k_in_bounds = comptime!(!this.config.check_col_bounds) || view_k < this.shape_k;
        m_in_bounds && k_in_bounds
    }
}

virtual_layout!(Im2colGlobalLayout, Im2colGlobalLayoutExpand);
