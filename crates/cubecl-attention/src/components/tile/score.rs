use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulSetupError;
use cubecl_matmul::components::tile::{TileConfig, TileMatmul, TileSetupInfo};

use crate::components::AttentionPrecision;
use crate::components::tile::shared::{KeyValue, ScoreProb};

#[cube]
pub trait ScoreMatmul<AP: AttentionPrecision>:
    TileMatmul<AP::ES, AP::ES, AP::EA, Rhs = KeyValue, Accumulator = ScoreProb<AP>>
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

pub struct ScoreTileMatmul;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScoreTileConfig;
impl TileConfig for ScoreTileConfig {
    fn plane_dim(&self) -> u32 {
        todo!()
    }

    fn matrix_layout(
        &self,
        ident: cubecl_matmul::components::StageIdent,
    ) -> cubecl_matmul::components::MatrixLayout {
        todo!()
    }

    fn stage_line_size(&self, ident: cubecl_matmul::components::StageIdent) -> u32 {
        todo!()
    }

    fn global_line_size(&self, ident: cubecl_matmul::components::StageIdent) -> u32 {
        todo!()
    }

    fn tile_size(&self) -> &cubecl_matmul::components::TileSize {
        todo!()
    }
}

impl<AP: AttentionPrecision> ScoreMatmul<AP> for ScoreTileMatmul {}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for ScoreTileMatmul {
    type Config = ScoreTileConfig;
    type Lhs = cmma::Matrix<L>;
    type Rhs = KeyValue<R>;
    type Accumulator = ScoreProb<A>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        let out = out.as_score();
        cmma::execute::<L, R, A, A>(lhs, rhs.as_key(), out, out);
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Lhs);
        unsafe {
            cmma::Matrix::<L>::uninitialized(
                cmma::MatrixIdent::A,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Rhs);
        unsafe {
            cmma::Matrix::<R>::uninitialized(
                cmma::MatrixIdent::B,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn fill_lhs<E: Numeric>(tile: &Tile<E>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Lhs, config);
        cmma::load(lhs, &slice, stride);
    }

    fn allocate_fill_cast_lhs<EI: Numeric>(
        tile: &Tile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Lhs {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Lhs, config);
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Lhs);
        let tmp = unsafe {
            cmma::Matrix::<EI>::uninitialized(
                cmma::MatrixIdent::A,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        };

        cmma::load(&tmp, &slice, stride);
        cmma::cast::<EI, L>(&tmp)
    }

    fn fill_rhs<E: Numeric>(tile: &Tile<E>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Rhs, config);
        cmma::load(rhs, &slice, stride);
    }

    fn fill_accumulator(
        tile: &Tile<A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Acc, config);
        cmma::load_with_layout(acc, &slice, stride, layout);
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<A, E>(out);
        cmma::store(
            slice,
            &acc,
            config.tile_size().n(),
            cmma::MatrixLayout::RowMajor,
        );
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.tile_size();
        unsafe {
            cmma::Matrix::<A>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, A::from_int(0));
    }
}
