use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{
    MatrixLayout,
    stage::ReaderFamily,
    tile::{TileMatmulFamily, TileSetupInfo},
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    stage::{
        AttentionTilingLayout, StageAttentionFamily,
        dummy::{AttentionStageMemoryConfig, DummyStageAttention, config::DummyStageConfig},
    },
};

pub struct DummyStageAttentionFamily<STM: TileMatmulFamily, VTM: TileMatmulFamily, RF: ReaderFamily>
{
    _phantom: PhantomData<(STM, VTM, RF)>,
}

impl<STM: TileMatmulFamily, VTM: TileMatmulFamily, RF: ReaderFamily> StageAttentionFamily
    for DummyStageAttentionFamily<STM, VTM, RF>
{
    type Attention<AP: AttentionPrecision> = DummyStageAttention<
        AP,
        STM::Matmul<AP::MatmulPrecision>,
        VTM::Matmul<AP::MatmulPrecision>,
        RF::Reader<AP::ES, AttentionTilingLayout>,
    >;

    type Config = DummyStageConfig<STM::Config, VTM::Config>;

    type KeyReader = RF;
    type ValueReader = RF;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        _problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let score_tile_config = STM::setup::<AP::MatmulPrecision, R>(
            client,
            score_tile_matmul_setup_info(selection, line_sizes),
        )?;
        let value_tile_config = VTM::setup::<AP::MatmulPrecision, R>(
            client,
            value_tile_matmul_setup_info(selection, line_sizes),
        )?;

        DummyStageConfig::new(
            AttentionStageMemoryConfig::new(score_tile_config),
            AttentionStageMemoryConfig::new(value_tile_config),
            1,
        )
    }
}

fn score_tile_matmul_setup_info(
    selection: &AttentionSelection,
    line_sizes: &AttentionLineSizes,
) -> TileSetupInfo {
    TileSetupInfo {
        tile_size: selection.score_tile_size,
        plane_dim: selection.plane_dim,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        lhs_line_size: line_sizes.query as u32,
        rhs_line_size: line_sizes.key as u32,
        // Not sure the out line size
        out_line_size: line_sizes.key as u32,
    }
}

fn value_tile_matmul_setup_info(
    selection: &AttentionSelection,
    line_sizes: &AttentionLineSizes,
) -> TileSetupInfo {
    TileSetupInfo {
        tile_size: selection.value_tile_size,
        plane_dim: selection.plane_dim,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        // Not sure
        lhs_line_size: line_sizes.value as u32,
        rhs_line_size: line_sizes.value as u32,
        out_line_size: line_sizes.value as u32,
    }
}
