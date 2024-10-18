use crate::matmul::cmma_matmul::config::StageDim;
use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_stage::{SmmConfig, StageMatmul};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaStageMatmulConfig<T: TmmConfig> {
    tmm_config: T,
    lhs_stage_dim: StageDim,
    rhs_stage_dim: StageDim,
    out_stage_dim: StageDim,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    num_planes: u32,
}

impl<T: TmmConfig> ComptimeConfig for CmmaStageMatmulConfig<T> {}

impl<T: TmmConfig> SmmConfig for CmmaStageMatmulConfig<T> {
    type TmmConfig = T;

    fn into_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn stage_dim(&self, ident: Ident) -> StageDim {
        match ident {
            Ident::Lhs => self.lhs_stage_dim,
            Ident::Rhs => self.rhs_stage_dim,
            Ident::Out => self.out_stage_dim,
        }
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }
}

impl<T: TmmConfig> MatmulConfig for CmmaStageMatmulConfig<T> {}

impl<T: TmmConfig> CmmaStageMatmulConfig<T> {
    pub fn new(
        lhs_stage_dim: StageDim,
        rhs_stage_dim: StageDim,
        out_stage_dim: StageDim,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        num_planes: u32,
        tmm_config: T,
    ) -> Self {
        Self {
            tmm_config,
            lhs_stage_dim,
            rhs_stage_dim,
            out_stage_dim,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
            lhs_layout,
            rhs_layout,
            num_planes,
        }
    }

    // pub fn from_stage_matmul<
    //     I: Numeric,
    //     O: Numeric,
    //     SMM: StageMatmul<
    //         I: Numeric,
    //         O: Numeric,
    //         Lhs: StageReader<I, S>,
    //         Rhs: StageReader<I, S>,
    //         Out: StageWriter<O>,
    //         Self,
    //     >,
    // >() {
    // }
}

// impl<I, O, Acc, TMM, StageSize, T>
//     CmmaStageMatmul<I, O, Acc, TMM, StageSize, CmmaStageMatmulConfig<T>>
// where
//     I: Numeric,
//     O: Numeric,
//     Acc: Numeric,
//     TMM: TileMatmul<I, Acc, Config = T>,
//     StageSize: CmmaStageSize,
//     T: TmmConfig,
// {
//     pub fn make_config(
//         cube_dim: &CubeDim,
//         cube_count: &CubeCount,
//         problem: &MatmulProblem,
//     ) -> CmmaStageMatmulConfig<T> {
//         CmmaStageMatmulConfig {
//             tmm_config: T::default(cube_dim, cube_count, problem),
//             lhs_stage_dim: StageDim {
//                 tile_size_x: TMM::M,
//                 tile_size_y: TMM::K,
//                 num_tiles_x: Self::M / TMM::M,
//                 num_tiles_y: Self::K / TMM::K,
//             },
//             rhs_stage_dim: StageDim {
//                 tile_size_x: TMM::K,
//                 tile_size_y: TMM::N,
//                 num_tiles_x: Self::K / TMM::K,
//                 num_tiles_y: Self::N / TMM::N,
//             },
//             out_stage_dim: StageDim {
//                 tile_size_x: TMM::M,
//                 tile_size_y: TMM::N,
//                 num_tiles_x: Self::M / TMM::M,
//                 num_tiles_y: Self::N / TMM::N,
//             },
//         }
//     }
// }
