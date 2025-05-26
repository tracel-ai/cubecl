use crate::matmul::components::{
    CompleteStageTiling, Ident, InputIdent, MatmulConfig, MatmulPrecision, MatmulSize,
    MatrixLayout, TilingDimensions,
    global::AccumulatorLoader,
    stage::{StageBuffering, StageConfig},
    tile::{TileConfig, TileMatmul},
};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct CommonStageConfig<T: TileConfig> {
    pub tmm_config: T,
    pub tiling: CompleteStageTiling,
    pub num_planes: u32,
    pub quantized: bool,
    pub buffering: StageBuffering,
    pub num_stages: NumStages,
    pub accumulator_count: AccumulatorCount,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageVectorization {
    /// A line size of zero means use the same vectorization as global memory.
    pub stage_line_size: u8,
    /// Still unsupported.
    pub stage_elem_padding: u8,
}

impl<T: TileConfig> StageConfig for CommonStageConfig<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn stage_line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.stage_line_size(ident)
    }

    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions {
        self.tiling.get(ident)
    }

    fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
        self.tmm_config.matrix_layout(ident)
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tmm_config.plane_dim()
    }

    fn tile_count(&self) -> &MatmulSize {
        &self.tiling.tile_count
    }

    fn buffering(&self) -> StageBuffering {
        self.buffering
    }

    fn num_stages(&self, ident: InputIdent) -> u32 {
        match ident {
            InputIdent::Lhs => self.num_stages.lhs,
            InputIdent::Rhs => self.num_stages.rhs,
        }
    }

    fn accumulator_count(&self) -> AccumulatorCount {
        self.accumulator_count
    }
}

impl<T: TileConfig> MatmulConfig for CommonStageConfig<T> {}

impl<T: TileConfig> CommonStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tmm_config: T,
        tiling: CompleteStageTiling,
        num_planes: u32,
        quantized: bool,
        buffering: StageBuffering,
        num_stages: NumStages,
        accumulator_count: AccumulatorCount,
    ) -> Self {
        Self {
            tmm_config,
            tiling,
            num_planes,
            quantized,
            buffering,
            num_stages,
            accumulator_count,
        }
    }
}

#[derive(CubeType)]
/// Wrapper over a sequence of tile matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<MP: MatmulPrecision, TMM: TileMatmul<MP>> {
    sequence: Sequence<TMM::Accumulator>,
    #[cube(comptime)]
    pub shape: AccumulatorCount,
}

#[cube]
impl<MP: MatmulPrecision, TMM: TileMatmul<MP>> Accumulators<MP, TMM> {
    pub fn new(#[comptime] config: CommonStageConfig<TMM::Config>) -> Accumulators<MP, TMM> {
        let shape = config.accumulator_count();
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(shape.m * shape.n) {
            accumulators.push(TMM::allocate_accumulator(config.to_tmm_config()));
        }

        Accumulators::<MP, TMM> {
            sequence: accumulators,
            shape,
        }
    }

    pub fn zero(&mut self, #[comptime] config: CommonStageConfig<TMM::Config>) {
        #[unroll]
        for i in 0..comptime![self.shape.num_tiles()] {
            TMM::zero_accumulator(self.sequence.index_mut(i), config.to_tmm_config());
        }
    }

    pub fn fill<L: AccumulatorLoader<MP>>(
        &mut self,
        loader: &mut L,
        #[comptime] config: CommonStageConfig<TMM::Config>,
    ) {
        #[unroll]
        for i in 0..comptime![self.shape.num_tiles()] {
            let acc = self.sequence.index_mut(i);
            L::load::<TMM>(loader, acc, i, config.to_tmm_config());
        }
    }

    pub fn get_at(
        this: &Accumulators<MP, TMM>,
        #[comptime] i: u32,
        #[comptime] j: u32,
    ) -> &TMM::Accumulator {
        this.sequence.index(comptime!(i * this.shape.n + j))
    }

    pub fn get_at_mut(
        this: &mut Accumulators<MP, TMM>,
        #[comptime] i: u32,
        #[comptime] j: u32,
    ) -> &mut TMM::Accumulator {
        this.sequence.index_mut(comptime!(i * this.shape.n + j))
    }
}

#[derive(CubeType)]
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NumStages {
    lhs: u32,
    rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AccumulatorCount {
    pub m: u32,
    pub n: u32,
}

impl From<(u32, u32)> for AccumulatorCount {
    fn from(value: (u32, u32)) -> Self {
        AccumulatorCount {
            m: value.0,
            n: value.1,
        }
    }
}

impl AccumulatorCount {
    pub fn num_tiles(&self) -> u32 {
        self.m * self.n
    }
}
