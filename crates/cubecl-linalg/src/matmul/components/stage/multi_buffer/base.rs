use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::stage::shared::{
    stage_matmul_size, CommonStageConfig, CommonStageInput,
};
use crate::matmul::components::stage::{StageMatmul, StageMatmulFamily};
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::{
    components::{
        global,
        stage::{StageConfig as _, StageWriter},
        tile, Ident, MatmulProblem,
    },
    kernels::matmul::{create_stage_dim, AdvancedConfig},
};

use super::reader::{LhsReader, RhsReader};
use super::{LhsReaderFamily, RhsReaderFamily};

pub struct MultiBufferMatmulFamily<TMM: TileMatmulFamily> {
    _instruction: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> StageMatmulFamily for MultiBufferMatmulFamily<TMM> {
    fn size(config: &Self::Config) -> MatmulSize {
        let tmm_config = config.to_tmm_config();
        stage_matmul_size::<TMM>(&tmm_config, &config.num_stage)
    }

    fn num(config: &Self::Config) -> MatmulSize {
        config.num_stage
    }

    type LhsReader = LhsReaderFamily;
    type RhsReader = RhsReaderFamily;
    type Matmul<I: Numeric, O: Numeric, Acc: Numeric> =
        MultiBufferMatmul<I, O, Acc, TMM::Matmul<I, Acc>>;
}

impl<TMM: TileMatmulFamily> MatmulConfigFactory for MultiBufferMatmulFamily<TMM> {
    type Input = CommonStageInput<TMM>;
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        check_num_planes(
            config.stage_dim(Ident::Lhs).tile_count_row(),
            config.num_planes(),
        )?;
        TMM::check_config(&config.to_tmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tmm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let tile = input.tile;
        let tmm_config = TMM::make_config(tile, problem, cube_dim, cube_count, advanced_config);
        let tmm_size = TMM::size(&tmm_config);
        let stage_size = stage_matmul_size::<TMM>(&tmm_config, &input.num_stages);

        let (tile_m, tile_n, tile_k) = (tmm_size.m, tmm_size.n, tmm_size.k);
        let (lhs_stage_dim, rhs_stage_dim, out_stage_dim) = create_stage_dim(
            stage_size.m,
            stage_size.n,
            stage_size.k,
            tile_m,
            tile_n,
            tile_k,
        );

        CommonStageConfig::new(
            tmm_config,
            input.num_stages,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            cube_dim.y,
            advanced_config.lhs_tiling_order,
            advanced_config.rhs_tiling_order,
        )
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct MultiBufferMatmul<I: Numeric, O: Numeric, EA: Numeric, TMM: tile::TileMatmul<I, EA>> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<EA>,
    _instruction: PhantomData<TMM>,
}

#[cube]
impl<I, O, EA, TMM> StageMatmul<I, O, EA> for MultiBufferMatmul<I, O, EA, TMM>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: tile::TileMatmul<I, EA>,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = LhsReader<I>;
    type RhsReader = RhsReader<I>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsReader<I>,
        rhs_reader: &RhsReader<I>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for buffer_iter in 0..config.num_stage.k {
            let tile_lhs =
                LhsReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, buffer_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_tile, config.to_tmm_config());

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = RhsReader::read_tile::<TMM::Config>(
                    rhs_reader,
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                TMM::fill_rhs(&tile_rhs, rhs_tile, config.to_tmm_config());

                let accumulator = acc.index_mut(accumulator_iter);
                TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());
            }
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            TMM::allocate_rhs(config.to_tmm_config()),
        )
    }

    fn read_accumulator<SW: StageWriter<O>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines = stage_config.stage_dim(Ident::Out).tile_size() / out_smem_line_size;

        let start = num_tile_lines * UNIT_POS_Y;
        let mut out_smem = SharedMemory::<O>::new_lined(
            num_tile_lines * stage_config.num_planes(),
            out_smem_line_size,
        );

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let mut smem_slice = out_smem.slice_mut(start, start + num_tile_lines);
            TMM::read_accumulator(accumulator, &mut smem_slice, stage_config.to_tmm_config());
            SW::write::<O, G>(
                out,
                smem_slice.to_slice(),
                UNIT_POS_Y,
                accumulator_iter,
                global_config,
            );
        }
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut acc = Sequence::<TMM::Accumulator>::new();

        #[unroll]
        for _ in 0..config.num_stage.n {
            acc.push(TMM::allocate_accumulator(config.to_tmm_config()));
        }

        acc
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        #[unroll]
        for i in 0..config.num_stage.n {
            TMM::zero_accumulator(acc.index_mut(i), config.to_tmm_config());
        }
    }

    fn fill_accumulator<L: AccumulatorLoader<O, EA, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for i in 0..config.num_stage.n {
            let acc = acc.index_mut(i);
            L::load::<I, TMM>(loader, acc, i, config.to_tmm_config());
        }
    }
}

fn check_num_planes(
    expected_num_planes: u32,
    actual_num_planes: u32,
) -> Result<(), InvalidConfigError> {
    if expected_num_planes != actual_num_planes {
        return Err(Box::new("Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}."));
    }

    Ok(())
}
