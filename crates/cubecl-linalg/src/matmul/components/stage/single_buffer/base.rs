use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::stage::{MatmulFamily, StageSize, TilingOrderConfig};
use crate::matmul::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::matmul::components::{LhsStageDim, OutStageDim, RhsStageDim};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::{
    components::{
        config::MatmulConfig,
        global::{self, AccumulatorLoader},
        stage::{self, Config as _, StageWriter},
        Ident, MatmulConfigFactory, MatmulProblem, MatrixLayout, StageDim,
    },
    kernels::matmul::{create_stage_dim, AdvancedConfig},
};

use super::{LhsBufferReader, LhsBufferReaderFamily, RhsBufferReader, RhsBufferReaderFamily};

pub struct SingleBufferMatmulFamily<TMM: TileMatmulFamily, SS: StageSize> {
    _instruction: PhantomData<TMM>,
    _block_size: PhantomData<SS>,
}

impl<TMM: TileMatmulFamily, SS: StageSize> MatmulFamily for SingleBufferMatmulFamily<TMM, SS> {
    const M: u32 = SS::NUM_M * TMM::M;
    const N: u32 = SS::NUM_N * TMM::N;
    const K: u32 = SS::NUM_K * TMM::K;

    type LhsReader = LhsBufferReaderFamily;
    type RhsReader = RhsBufferReaderFamily;
    type Matmul<I: Numeric, O: Numeric, Acc: Numeric> =
        SingleBufferMatmul<I, O, Acc, TMM::Matmul<I, Acc>, SS>;
}

impl<TMM, SS> MatmulConfigFactory for SingleBufferMatmulFamily<TMM, SS>
where
    TMM: TileMatmulFamily,
    SS: StageSize,
{
    type Config = Config<TMM::Config>;

    fn check_config(config: Self::Config) {
        TMM::check_config(config.to_tmm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
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
            lhs_stage_dim.num_tiles_x_dim(),
            advanced_config.lhs_tiling_order,
            advanced_config.rhs_tiling_order,
        )
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// Very similar to multi buffer, except is unable to have more than one buffer, and takes BufferReaders for StageReaders
///
/// # Assumptions
/// - There are at least as many planes as the stage size in m
pub struct SingleBufferMatmul<
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
    SS: StageSize,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<EA>,
    _instruction: PhantomData<TMM>,
    _block_size: PhantomData<SS>,
}

#[cube]
impl<I, O, EA, TMM, SS> stage::Matmul<I, O, EA> for SingleBufferMatmul<I, O, EA, TMM, SS>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
    SS: StageSize,
{
    type Config = Config<TMM::Config>;

    type LhsReader = LhsBufferReader<I>;
    type RhsReader = RhsBufferReader<I>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsBufferReader<I>,
        rhs_reader: &RhsBufferReader<I>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let tile_lhs = LhsBufferReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, config);
        TMM::fill_lhs(&tile_lhs, lhs_tile, config.to_tmm_config());

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let tile_rhs =
                RhsBufferReader::read_tile::<TMM::Config>(rhs_reader, accumulator_iter, config);
            TMM::fill_rhs(&tile_rhs, rhs_tile, config.to_tmm_config());

            let accumulator = acc.index_mut(accumulator_iter);
            TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::init_lhs(config.to_tmm_config()),
            TMM::init_rhs(config.to_tmm_config()),
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut accumulators = Sequence::<TMM::Accumulator>::new();

        #[unroll]
        for _ in 0..SS::NUM_N {
            accumulators.push(TMM::init_accumulator(config.to_tmm_config()));
        }

        accumulators
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
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct Config<T: TileConfig> {
    tmm_config: T,
    lhs_stage_dim: LhsStageDim,
    rhs_stage_dim: RhsStageDim,
    out_stage_dim: OutStageDim,
    num_planes: u32,
    lhs_tiling_order: TilingOrderConfig,
    rhs_tiling_order: TilingOrderConfig,
}

impl<T: TileConfig> stage::Config for Config<T> {
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

impl<T: TileConfig> MatmulConfig for Config<T> {}

impl<T: TileConfig> Config<T> {
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
