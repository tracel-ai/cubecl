use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use cubecl_matmul::components::{MatrixPrecision, stage::StageMemoryConfig};
use cubecl_std::FastDivmod;

use crate::components::{
    ConvolutionParams, Dimensionality,
    global::{args::RuntimeArgs, memory::Im2colTmaReader},
};
use cubecl_matmul::components::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, StridedStageMemory,
};

pub type TmaIm2colTiling = ContiguousTilingLayout<ColMajorTilingOrder>;
pub type TmaIm2colStage<IP> = StridedStageMemory<<IP as MatrixPrecision>::Stage, TmaIm2colTiling>;

/// Reader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct TmaIm2colGlobalReader<IP: MatrixPrecision> {
    pub map: Im2colTmaReader<IP::Global>,
    pub stages: Sequence<StridedStageMemory<IP::Stage, TmaIm2colTiling>>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    params: ConvolutionParams,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> TmaIm2colGlobalReader<IP> {
    pub fn new(
        tensor: TensorMap<Line<IP::Global>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] num_stages: u32,
        #[comptime] params: ConvolutionParams,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        let mut stages = Sequence::new();

        #[unroll]
        for _ in 0..num_stages {
            stages.push(StridedStageMemory::new_aligned(128u32, config))
        }

        let (n_offs, spatial_offsets) = div_mod_seq(x_offset, &runtime_args.shape_out);

        let map = Im2colTmaReader::<IP::Global>::new(tensor, n_offs, spatial_offsets, y_offset);

        TmaIm2colGlobalReader::<IP> {
            map,
            stages,
            padded_channels: runtime_args.padded_channels,
            params,
            config,
        }
    }

    pub fn fill_stage(&mut self, bar: &Barrier, #[comptime] stage_idx: u32) {
        let stage = self.stages.index_mut(stage_idx);
        let params = comptime![self.params];
        let config = comptime![self.config];

        if UNIT_POS == 0 {
            let m_size = config.elements_per_stage_along_row();
            let k_size = config.elements_per_tile_along_col;
            let slice_size = m_size * k_size;
            let mut full_stage = stage.as_slice_mut(1u32);
            let tensor = self.map.tensor.try_cast_unchecked();

            let spatial_dims = comptime![self.map.spatial_offsets.len()];
            let mut in_offs = Sequence::<i32>::new();

            #[unroll]
            for dim in 0..spatial_dims {
                let offs =
                    self.map.spatial_offsets.index(dim) * comptime![params.stride[dim as usize]];
                let offs = offs as i32 - comptime![params.padding[dim as usize]];
                in_offs.push(offs);
            }

            #[unroll]
            for tile_k in 0..config.tiles_per_stage_along_col() {
                let k = self.map.k_offset + tile_k * k_size;
                let (k_idx, channel_start) = self.padded_channels.div_mod(k);
                let slice_start = tile_k * slice_size;
                let mut stage = full_stage.slice_mut(slice_start, slice_start + slice_size);

                match params.dimensionality {
                    Dimensionality::Dim1 => {
                        let offset = k_idx * comptime![params.dilation[0]];

                        bar.tma_load_im2col_3d(
                            &tensor,
                            &mut stage,
                            self.map.n_offset as i32,
                            *in_offs.index(0),
                            channel_start as i32,
                            offset as u16,
                        );
                    }
                    Dimensionality::Dim2 => {
                        let (k_x, k_y) = (
                            k_idx % comptime![params.kernel_size[1]],
                            k_idx / comptime![params.kernel_size[1]],
                        );

                        let offset_y = k_y * comptime![params.dilation[0]];
                        let offset_x = k_x * comptime![params.dilation[1]];

                        bar.tma_load_im2col_4d(
                            &tensor,
                            &mut stage,
                            self.map.n_offset as i32,
                            *in_offs.index(0),
                            *in_offs.index(1),
                            channel_start as i32,
                            offset_y as u16,
                            offset_x as u16,
                        );
                    }
                    Dimensionality::Dim3 => {
                        let (k_x, rem) = (
                            k_idx % comptime![params.kernel_size[2]],
                            k_idx / comptime![params.kernel_size[2]],
                        );
                        let (k_y, k_z) = (
                            rem % comptime![params.kernel_size[1]],
                            rem / comptime![params.kernel_size[1]],
                        );

                        let offset_z = k_z * comptime![params.dilation[0]];
                        let offset_y = k_y * comptime![params.dilation[1]];
                        let offset_x = k_x * comptime![params.dilation[2]];

                        bar.tma_load_im2col_5d(
                            &tensor,
                            &mut stage,
                            self.map.n_offset as i32,
                            *in_offs.index(0),
                            *in_offs.index(1),
                            *in_offs.index(2),
                            channel_start as i32,
                            offset_z as u16,
                            offset_y as u16,
                            offset_x as u16,
                        );
                    }
                }
            }
        }
    }

    pub fn advance_view(&mut self, k_offset: u32) {
        self.map.update_view(k_offset);
    }

    pub fn stage(&self, #[comptime] stage_idx: u32) -> TmaIm2colStage<IP> {
        *self.stages.index(stage_idx)
    }
}

/// Decompose a linear index into local positions along each dimension in `shape`. Also returns the
/// left over remainder.
#[cube]
pub(crate) fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}
