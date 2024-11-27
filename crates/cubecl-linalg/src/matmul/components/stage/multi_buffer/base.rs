use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::stage::{StageSize, TilingOrderConfig};
use crate::matmul::components::{global::AccumulatorLoader, stage::base::Matmul as _};
use crate::matmul::components::{LhsStageDim, OutStageDim, RhsStageDim};
use crate::matmul::{
    components::{
        config::MatmulConfig,
        global,
        stage::{self, Config as _, StageWriter},
        tile, Ident, MatmulKernel, MatmulProblem, MatrixLayout, StageDim,
    },
    kernels::matmul::{create_stage_dim, AdvancedConfig},
};

use super::reader::{LhsReader, RhsReader};

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct Matmul<I: Numeric, O: Numeric, EA: Numeric, TMM: tile::Matmul<I, EA>, SS: StageSize> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<EA>,
    _instruction: PhantomData<TMM>,
    _block_size: PhantomData<SS>,
}

#[cube]
impl<I, O, EA, TMM, SS> stage::Matmul<I, O, EA> for Matmul<I, O, EA, TMM, SS>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: tile::Matmul<I, EA>,
    SS: StageSize,
{
    const M: u32 = SS::NUM_M * TMM::M;
    const N: u32 = SS::NUM_N * TMM::N;
    const K: u32 = SS::NUM_K * TMM::K;
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
        for buffer_iter in 0..SS::NUM_K {
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
            TMM::init_lhs(config.to_tmm_config()),
            TMM::init_rhs(config.to_tmm_config()),
        )
    }

    fn read_accumulator<SW: StageWriter<O>, G: global::Config>(
        acc: &Self::Accumulator,
        out: &mut SW,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.stage_dim(Ident::Out).tile_num_elements() / out_smem_line_size;

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
        for _ in 0..SS::NUM_N {
            acc.push(TMM::init_accumulator(config.to_tmm_config()));
        }

        acc
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        #[unroll]
        for i in 0..SS::NUM_N {
            TMM::zero_accumulator(acc.index_mut(i), config.to_tmm_config());
        }
    }

    fn fill_accumulator<L: AccumulatorLoader<O, EA, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for i in 0..SS::NUM_N {
            let acc = acc.index_mut(i);
            L::load::<I, TMM>(loader, acc, i, config.to_tmm_config());
        }
    }
}

impl<I, O, Acc, TMM, SS> MatmulKernel<I, O> for Matmul<I, O, Acc, TMM, SS>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    TMM: tile::Matmul<I, Acc>,
    SS: StageSize,
{
    type Config = Config<TMM::Config>;

    fn check_config(config: Self::Config) {
        comptime!(check_num_planes(
            config.stage_dim(Ident::Lhs).num_tiles_x_dim(),
            config.num_planes()
        ));
        TMM::check_config(config.to_tmm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
        TMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let tmm_config = TMM::make_config(problem, cube_dim, cube_count, advanced_config);

        let (stage_m, stage_n, stage_k) = (Self::M, Self::N, Self::K);
        let (tile_m, tile_n, tile_k) = (TMM::M, TMM::N, TMM::K);
        let (lhs_stage_dim, rhs_stage_dim, out_stage_dim) =
            create_stage_dim(stage_m, stage_n, stage_k, tile_m, tile_n, tile_k);

        Config::new(
            tmm_config,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            cube_dim.y,
            advanced_config.lhs_tiling_order,
            advanced_config.rhs_tiling_order,
        )
    }
}

fn check_num_planes(expected_num_planes: u32, actual_num_planes: u32) {
    assert_eq!(
        expected_num_planes, actual_num_planes,
        "Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}.",
    );
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the multi buffer matmul
pub struct Config<T: tile::Config> {
    tmm_config: T,
    lhs_stage_dim: LhsStageDim,
    rhs_stage_dim: RhsStageDim,
    out_stage_dim: OutStageDim,
    num_planes: u32,
    lhs_tiling_order: TilingOrderConfig,
    rhs_tiling_order: TilingOrderConfig,
}

impl<T: tile::Config> stage::Config for Config<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim> {
        match ident {
            Ident::Lhs => Box::new(self.lhs_stage_dim),
            Ident::Rhs => Box::new(self.rhs_stage_dim),
            Ident::Out => Box::new(self.out_stage_dim),
        }
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        self.tmm_config.layout(ident)
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tmm_config.plane_dim()
    }

    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
        match ident.as_input() {
            InputIdent::Lhs => self.lhs_tiling_order,
            InputIdent::Rhs => self.rhs_tiling_order,
        }
    }
}

impl<T: tile::Config> MatmulConfig for Config<T> {}

impl<T: tile::Config> Config<T> {
    pub fn new(
        tmm_config: T,
        lhs_stage_dim: LhsStageDim,
        rhs_stage_dim: RhsStageDim,
        out_stage_dim: OutStageDim,
        num_planes: u32,
        lhs_tiling_order: TilingOrderConfig,
        rhs_tiling_order: TilingOrderConfig,
    ) -> Self {
        Self {
            tmm_config,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            num_planes,
            lhs_tiling_order,
            rhs_tiling_order,
        }
    }
}
