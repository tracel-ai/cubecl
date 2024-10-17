use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::CmmaStageSize;
use crate::matmul::cmma_matmul::config::{CmmaConfig, CmmaPreConfig};
use crate::matmul::config::PlaneMapper;
use crate::matmul::launch::stage_matmul_launch;
use crate::matmul::matmul_stage::{SmmConfig, StageMatmul, StageReader, StageWriter};
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matrix::Ident;
use crate::matmul::Matmul;

pub struct CmmaStageMatmul<
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc>,
    BlockSize: CmmaStageSize,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<Acc>,
    _instruction: PhantomData<Tmm>,
    _block_size: PhantomData<BlockSize>,
}

#[cube]
impl<I, O, Acc, Tmm, StageSize, Lhs, Rhs, Out> StageMatmul<I, O, Lhs, Rhs, Out>
    for CmmaStageMatmul<I, O, Acc, Tmm, StageSize>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc, Config = CmmaConfig>,
    StageSize: CmmaStageSize,
    Lhs: StageReader<I, Config = Self::Config>,
    Rhs: StageReader<I, Config = Self::Config>,
    Out: StageWriter<O, Config = Self::Config>,
{
    const M: u32 = StageSize::M;
    const N: u32 = StageSize::N;
    const K: u32 = StageSize::K;
    type Accumulator = Sequence<Tmm::Out>;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator, config: Self::Config) {
        let num_buffers = StageSize::K / Tmm::K;

        let mut instruction_lhs = Tmm::init_lhs(comptime!(config.layout(Ident::Lhs)));
        let mut instruction_rhs = Tmm::init_rhs(comptime!(config.layout(Ident::Rhs)));

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let tile_lhs = Lhs::read_tile(&lhs, Self::plane_id(), buffer_iter, 0u32, config);
            Tmm::fill_lhs(tile_lhs, &mut instruction_lhs);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = Rhs::read_tile(
                    &rhs,
                    Self::plane_id(),
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                Tmm::fill_rhs(tile_rhs, &mut instruction_rhs);

                let accumulator = acc.index_mut(accumulator_iter);
                Tmm::execute(&instruction_lhs, &instruction_rhs, accumulator);
            }
        }
    }

    fn acc_init_zeros() -> Self::Accumulator {
        let mut accumulators = Sequence::<Tmm::Out>::new();
        let num_accumulators = StageSize::N / Tmm::N;

        #[unroll]
        for _ in 0..num_accumulators {
            accumulators.push(Tmm::init_output());
        }

        accumulators
    }

    fn acc_read(acc: &Self::Accumulator, out: &mut Out, #[comptime] config: Self::Config) {
        let line_size = config.out_smem_line_size;
        let num_tile_lines = config.tile_num_elems(Ident::Out) / line_size;
        let start = num_tile_lines * Self::plane_id();

        let mut smem =
            SharedMemory::<O>::new_lined(num_tile_lines * comptime!(config.num_planes), line_size);

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let smem_slice = smem.slice_mut(start, start + num_tile_lines);
            Tmm::read_output(accumulator, smem_slice);
            Out::write(
                out,
                &smem_slice.as_slice(),
                Self::plane_id(),
                accumulator_iter,
                line_size,
                config,
            );
        }
    }
}

impl<I, O, Acc, TMM, StageSize> Matmul<I, O> for CmmaStageMatmul<I, O, Acc, TMM, StageSize>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    TMM: TileMatmul<I, Acc, Config = CmmaConfig>,
    StageSize: CmmaStageSize,
{
    type Config = CmmaConfig;

    fn preconfigure() -> CmmaPreConfig {
        let mut pre_config = TMM::preconfigure();

        pre_config.lhs_num_tiles_x = Some(StageSize::M / TMM::M);
        pre_config.lhs_num_tiles_y = Some(StageSize::K / TMM::K);

        pre_config.rhs_num_tiles_x = Some(StageSize::K / TMM::K);
        pre_config.rhs_num_tiles_y = Some(StageSize::N / TMM::N);

        pre_config.out_num_tiles_x = Some(StageSize::M / TMM::M);
        pre_config.out_num_tiles_y = Some(StageSize::N / TMM::N);

        pre_config
    }

    fn check_config(config: Self::Config) {
        let _ = comptime!(check_num_planes(
            config.stage_dims.lhs.num_tiles_x,
            config.num_planes
        ));
        TMM::check_config(config.into_tmm_config());
    }

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::Config,
    ) {
        Self::check_config(config);
        stage_matmul_launch::launch_unchecked::<I, O, Self, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}

#[cube]
impl<I, O, Acc, Tmm, Block> PlaneMapper for CmmaStageMatmul<I, O, Acc, Tmm, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc>,
    Block: CmmaStageSize,
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

fn check_num_planes(expected_num_planes: u32, actual_num_planes: u32) {
    assert_eq!(
        expected_num_planes, actual_num_planes,
        "Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}.",
    );
}
