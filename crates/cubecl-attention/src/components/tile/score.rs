use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::{Tile, TileConfig, TileMatmul, TileSetupInfo};
use cubecl_matmul::components::{LhsS, MatmulPrecision, MatmulSetupError, RhsS};

use crate::components::AttentionPrecision;
use crate::components::tile::shared::{KeyValue, ScoreProb};
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::StageIdent;

use cubecl_matmul::components::TileSize;

#[cube]
pub trait ScoreMatmul<AP: AttentionPrecision>:
    TileMatmul<AP::MatmulPrecision, Rhs = KeyValue, Accumulator = ScoreProb<AP>>
{
}

pub trait ScoreMatmulFamily: Send + Sync + 'static {
    type Matmul<AP: AttentionPrecision>: ScoreMatmul<AP>;
    type Config: TileConfig;

    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_setup_info: TileSetupInfo,
    ) -> Result<Self::Config, MatmulSetupError>;
}

// #[derive(CubeType)]
// pub struct DummyScoreMatmul<AP: AttentionPrecision> {
//     #[cube(comptime)]
//     _phantom: PhantomData<AP>,
// }
// impl<AP: AttentionPrecision> ScoreMatmul<AP> for DummyScoreMatmul<AP> {}

// #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// pub struct ScoreConfig {}
// impl TileConfig for ScoreConfig {
//     fn plane_dim(&self) -> u32 {
//         todo!()
//     }

//     fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
//         todo!()
//     }

//     fn stage_line_size(&self, ident: StageIdent) -> u32 {
//         todo!()
//     }

//     fn global_line_size(&self, ident: StageIdent) -> u32 {
//         todo!()
//     }

//     fn tile_size(&self) -> &TileSize {
//         todo!()
//     }
// }

// #[cube]
// impl<AP: AttentionPrecision, T: TileMatmul<AP::MatmulPrecision>> TileMatmul<AP::MatmulPrecision>
//     for DummyScoreMatmul<AP>
// {
//     type Config = ScoreConfig;
//     type Lhs = T::Lhs;
//     type Rhs;
//     type Accumulator;

//     fn execute(
//         lhs: &Self::Lhs,
//         rhs: &Self::Rhs,
//         out: &mut Self::Accumulator,
//         config: Self::Config,
//     ) {
//         todo!()
//     }

//     fn allocate_lhs(config: Self::Config) -> Self::Lhs {
//         todo!()
//     }

//     fn fill_lhs(tile: &Tile<LhsS<AP::MatmulPrecision>>, lhs: &mut Self::Lhs, config: Self::Config) {
//         todo!()
//     }

//     fn allocate_fill_cast_lhs<EI: Numeric>(tile: &Tile<EI>, config: Self::Config) -> Self::Lhs {
//         todo!()
//     }

//     fn allocate_rhs(config: Self::Config) -> Self::Rhs {
//         todo!()
//     }

//     fn fill_rhs(tile: &Tile<RhsS<AP::MatmulPrecision>>, rhs: &mut Self::Rhs, config: Self::Config) {
//         todo!()
//     }

//     fn allocate_accumulator(config: Self::Config) -> Self::Accumulator {
//         todo!()
//     }

//     fn fill_accumulator(
//         tile: &Tile<<AP::MatmulPrecision as MatmulPrecision>::EA>,
//         acc: &mut Self::Accumulator,
//         config: Self::Config,
//     ) {
//         todo!()
//     }

//     fn zero_accumulator(acc: &mut Self::Accumulator, config: Self::Config) {
//         todo!()
//     }

//     fn write_results(
//         out: &Self::Accumulator,
//         slice: &mut SliceMut<Line<<AP::MatmulPrecision as MatmulPrecision>::EO>>,
//         config: Self::Config,
//     ) {
//         todo!()
//     }
// }
