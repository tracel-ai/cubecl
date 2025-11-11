use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulIdent,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Coords3d, Layout, LayoutExpand},
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig, ConvolutionParams, ConvolutionProblem,
        global::{layout::NhwcCoords, read::im2col_tma::div_mod_seq},
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Maps a 4D NHWC tensor to a 2D column matrix using the im2col transformation
/// It first decomposes the `(m, k)` matrix into `((n, out_h, out_w), (k_h, k_w, c))`, then applies
/// the convolution parameters to calculate the position in the input tensor for that kernel element.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct Im2colLayout {
    /// Shape of output DHW
    pub shape_out: Sequence<FastDivmod>,
    /// Shape of channel, for decomposing k
    pub shape_channel: FastDivmod,

    /// Shape of the combined `m` dimension, including padding
    pub shape_m: u32,
    /// Shape of the combined `k` dimension, including padding
    pub shape_k: u32,

    /// Comptime parameters for the convolution
    #[cube(comptime)]
    pub params: ConvolutionParams,
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl Im2colLayout {
    pub fn new<G: GlobalConfig>(
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> Im2colLayout {
        let shape_out = args.shape_out.clone();

        Im2colLayout {
            shape_out,
            shape_channel: args.shape_channel,
            shape_m: args.shape_m,
            shape_k: args.shape_k,
            params: config.convolution_params(),
            config: config.global_memory_config(MatmulIdent::Lhs),
        }
    }
}

#[cube]
impl Layout for Im2colLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, pos: Self::Coordinates) -> NhwcCoords {
        let params = comptime![self.params];
        let (_, view_m, view_k) = pos;

        let (batch, out_offs) = div_mod_seq(view_m, &self.shape_out);

        let (mut rem, channel) = self.shape_channel.div_mod(view_k);

        let spatial_dims = comptime![self.shape_out.len()];
        let mut in_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![params.kernel_size[dim as usize]];
            let k_pos = rem % ksize;
            rem /= ksize;

            let out_pos = *out_offs.index(dim);
            let stride = comptime![params.stride[dim as usize]];
            let dilate = comptime![params.dilation[dim as usize]];
            let pad = comptime![params.padding[dim as usize]];

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

    fn shape(&self) -> Self::Coordinates {
        (1, self.shape_m, self.shape_k)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, view_m, view_k) = pos;
        // Shouldn't be relied on because it doesn't check spatial
        let m_in_bounds = comptime!(!self.config.check_row_bounds()) || view_m < self.shape_m;
        let k_in_bounds = comptime!(!self.config.check_col_bounds()) || view_k < self.shape_k;
        m_in_bounds && k_in_bounds
    }
}

impl<'a, R: Runtime> Im2colLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R::Server>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalMemoryConfig,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|s| FastDivmodArgs::new(client, *s as u32))
            .collect();
        let shape_channel = FastDivmodArgs::new(client, problem.channels as u32);

        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_k = ScalarArg::new(problem.k as u32);

        Im2colLayoutLaunch::new(shape_out, shape_channel, shape_m, shape_k, params, config)
    }
}
