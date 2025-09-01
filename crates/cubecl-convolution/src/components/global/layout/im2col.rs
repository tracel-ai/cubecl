use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_matmul::components::{
    MatmulIdent,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod,
    tensor::{
        layout::{Coords1d, Coords3d, Layout},
        r#virtual::VirtualTensor,
    },
};

use crate::{
    components::{
        ConvolutionConfig,
        global::{layout::virtual_layout, load::im2col_tma::div_mod_seq},
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D NHWC tensor to a 2D column matrix using the im2col transformation
/// It first decomposes the `(m, k)` matrix into `((n, out_h, out_w), (k_h, k_w, c))`, then applies
/// the convolution parameters to calculate the position in the input tensor for that kernel element.
#[derive(CubeType, Clone)]
pub struct Im2colGlobalLayout {
    /// Stride for N
    pub stride_batch: u32,
    /// Strides for DHW
    pub strides_spatial: Sequence<u32>,
    /// Stride for C
    pub stride_channel: u32,

    /// Shape of DHW
    pub shapes_spatial: Sequence<u32>,
    /// Shape of C
    pub shape_channel: u32,

    /// Shape of output DHW
    pub shape_out: Sequence<FastDivmod>,

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
    pub fn new<E: Numeric, G: GlobalConfig>(
        tensor: &VirtualTensor<E>,
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> Im2colGlobalLayout {
        let shape_out = args.shape_out.clone();
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

        Im2colGlobalLayout {
            stride_batch,
            strides_spatial,
            stride_channel,
            shapes_spatial,
            shape_channel,
            shape_out,
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
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        Self::to_source_pos_checked(this, coords).0
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (1, this.shape_m, this.shape_k)
    }

    fn to_source_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let (_, view_m, view_k) = coords;

        let (batch, out_offs) = div_mod_seq(view_m, &this.shape_out);

        let channel = view_k % this.shape_channel;
        let mut rem = view_k / this.shape_channel;

        let spatial_dims = comptime![this.shapes_spatial.len()];
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

        let has_padding = comptime! {
            let mut has_padding = false;
            for i in 0..spatial_dims {
                has_padding |= this.padding[i as usize] != 0;
            }
            has_padding
        };

        let m_in_bounds = comptime!(!this.config.check_row_bounds) || view_m < this.shape_m;
        let k_in_bounds = comptime!(!this.config.check_col_bounds) || view_k < this.shape_k;
        let mut spatial_in_bounds = true;

        // If padding != 0, needs to check bounds on each spatial dim to see if we're in padding
        // Without padding, checking `m` and `k` is enough.
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

        let line_size = this.config.global_line_size;

        (read_pos / line_size, in_bounds)
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

#[allow(unused_variables)]
#[cube]
pub(crate) fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}
