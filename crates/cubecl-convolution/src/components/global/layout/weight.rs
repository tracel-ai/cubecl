use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatmulElems,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Coords3d, Layout, LayoutExpand},
};

use crate::components::{
    ConvGemmConfig, ConvolutionConfig, ConvolutionParams, ConvolutionProblem,
    global::{args::RuntimeArgs, layout::NhwcCoords},
};

/// Maps a 4D weight tensor of shape `(out_c, (k_h, k_w, in_c))` to a col-major 2D matmul tile with
/// shape `(n, k)`
#[derive(CubeType, CubeLaunch, Clone)]
pub struct WeightLayout {
    /// Number of channels, including padding, used for decomposing `k`
    pub padded_channels: FastDivmod,

    /// Shape of the conceptual `k` size, including padding
    pub shape_k: u32,
    /// Shape of the conceptual `n` size, or `out_c`
    pub shape_n: u32,

    /// Size of the convolution kernel
    #[cube(comptime)]
    pub params: ConvolutionParams,
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl WeightLayout {
    pub fn new<E: Numeric, G: GlobalConfig>(
        args: &RuntimeArgs,
        #[comptime] config: ConvolutionConfig<G>,
    ) -> WeightLayout {
        WeightLayout {
            shape_k: args.shape_k,
            shape_n: args.shape_n,
            padded_channels: args.padded_channels,
            params: config.convolution_params,
            config: config.rhs_global_memory_config(),
        }
    }
}

#[cube]
impl Layout for WeightLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> NhwcCoords {
        let params = comptime![self.params];
        let (_, k, n) = coords;

        let (mut rem, in_c) = self.padded_channels.div_mod(k);

        let spatial_dims = comptime![params.dimensionality.num_dims()];
        let mut kernel_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![params.kernel_size[dim as usize]];
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
        (1, self.shape_k, self.shape_n)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, k, n) = pos;
        let check_k = comptime![self.config.check_row_bounds];
        let check_n = comptime![self.config.check_col_bounds];
        (!check_k || k < self.shape_k) && (!check_n || n < self.shape_n)
    }
}

impl<'a, R: Runtime> WeightLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalMemoryConfig,
        dtypes: &MatmulElems,
    ) -> Self {
        let load_width = client.properties().hardware.load_width;
        let channel_align = load_width / dtypes.lhs_global.size_bits() as u32;
        let padded_channels = (problem.channels as u32).next_multiple_of(channel_align);

        let size_k = problem.kernel_size.iter().product::<u32>() * padded_channels;
        let padded_channels = FastDivmodArgs::new(client, padded_channels);
        let shape_k = ScalarArg::new(size_k);
        let shape_n = ScalarArg::new(problem.n as u32);

        WeightLayoutLaunch::new(padded_channels, shape_k, shape_n, params, config)
    }
}
