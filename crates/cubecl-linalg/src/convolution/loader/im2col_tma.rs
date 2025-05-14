use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_core::{intrinsic, prelude::*};

use cubecl_std::{FastDivmod, tensor::r#virtual::VirtualTensor};
use std::marker::PhantomData;

use crate::{
    convolution::{
        ConvGemmConfig,
        base::{Dimensionality, RuntimeArgs},
        reader::tma::Im2colTmaReader,
    },
    matmul::components::{
        Ident, InputIdent, MatmulPrecision,
        stage::{
            ColMajorTilingOrder, ContiguousTilingLayout, FullStageToTileReader, StageConfig, StageMemory,
        },
    },
};

pub type TmaIm2colTiling = ContiguousTilingLayout<ColMajorTilingOrder>;
pub type TmaIm2colReader<MP> = FullStageToTileReader<<MP as MatmulPrecision>::ES, TmaIm2colTiling>;

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct TmaIm2colLoader<MP: MatmulPrecision, G: ConvGemmConfig> {
    pub map: Im2colTmaReader<MP::EI>,
    pub stages: Sequence<StageMemory<MP::ES, TmaIm2colTiling>>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: ConvGemmConfig> TmaIm2colLoader<MP, G> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] num_stages: u32,
        #[comptime] config: G,
    ) -> Self {
        let mut stages = Sequence::new();

        #[unroll]
        for _ in 0..num_stages {
            stages.push(StageMemory::new_aligned::<G::SmmConfig>(
                Ident::Lhs,
                128u32,
                config.to_smm_config(),
            ))
        }

        let (n_offs, spatial_offsets) = div_mod_seq(x_offset, &runtime_args.out_shape);

        let map = Im2colTmaReader::<MP::EI>::new(tensor, n_offs, spatial_offsets, y_offset);

        TmaIm2colLoader::<MP, G> {
            map,
            stages,
            padded_channels: runtime_args.padded_channels,
            _config: PhantomData::<G>,
        }
    }

    pub fn fill_stage(
        this: &mut Self,
        bar: &Barrier<MP::ES>,
        #[comptime] stage_idx: u32,
        #[comptime] config: G,
    ) {
        let tmm = config.to_smm_config();
        let tiling_dims = tmm.tiling_dimensions(Ident::Lhs);
        let stage = this.stages.index_mut(stage_idx);

        if UNIT_POS == 0 {
            let m_size = tiling_dims.total_row();
            let k_size = tiling_dims.tile_shape_col();
            let slice_size = m_size * k_size;
            let mut full_stage = stage.as_slice_mut(1u32);
            let tensor = this.map.tensor.try_cast_unchecked();

            let spatial_dims = comptime![this.map.spatial_offsets.len()];
            let mut in_offs = Sequence::<i32>::new();

            #[unroll]
            for dim in 0..spatial_dims {
                let dim = unwrap(dim);
                let offs = this.map.spatial_offsets.index(dim) * comptime![config.stride(dim)];
                let offs = offs as i32 - comptime![config.padding(dim)];
                in_offs.push(offs);
            }

            #[unroll]
            for tile_k in 0..tiling_dims.tile_count_col() {
                let k = this.map.k_offset + tile_k * k_size;
                let (k_idx, channel_start) = this.padded_channels.div_mod(k);
                let slice_start = tile_k * slice_size;
                let mut stage = full_stage.slice_mut(slice_start, slice_start + slice_size);

                match config.dimensionality() {
                    Dimensionality::Dim1 => {
                        let offset = k_idx * config.dilation(0);

                        bar.tma_load_im2col_3d(
                            &tensor,
                            &mut stage,
                            this.map.n_offset as i32,
                            *in_offs.index(0),
                            channel_start as i32,
                            offset as u16,
                        );
                    }
                    Dimensionality::Dim2 => {
                        let (k_x, k_y) =
                            (k_idx % config.kernel_size(1), k_idx / config.kernel_size(1));

                        let offset_y = k_y * config.dilation(0);
                        let offset_x = k_x * config.dilation(1);

                        bar.tma_load_im2col_4d(
                            &tensor,
                            &mut stage,
                            this.map.n_offset as i32,
                            *in_offs.index(0),
                            *in_offs.index(1),
                            channel_start as i32,
                            offset_y as u16,
                            offset_x as u16,
                        );
                    }
                    Dimensionality::Dim3 => {
                        let (k_x, rem) =
                            (k_idx % config.kernel_size(2), k_idx / config.kernel_size(2));
                        let (k_y, k_z) = (rem % config.kernel_size(1), rem / config.kernel_size(1));

                        let offset_z = k_z * config.dilation(0);
                        let offset_y = k_y * config.dilation(1);
                        let offset_x = k_x * config.dilation(2);

                        bar.tma_load_im2col_5d(
                            &tensor,
                            &mut stage,
                            this.map.n_offset as i32,
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

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.map.update_view(k_offset);
    }

    pub fn reader(this: &Self, #[comptime] stage_idx: u32) -> TmaIm2colReader<MP> {
        TmaIm2colReader::<MP>::new(*this.stages.index(stage_idx), InputIdent::Lhs)
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
        let i = unwrap(i);
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}

#[allow(unused_variables)]
#[cube]
fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}
