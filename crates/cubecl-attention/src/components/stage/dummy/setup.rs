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

pub struct DummyStageAttentionFamily<SM: TileMatmulFamily, VM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(SM, VM, RF)>,
}

impl<SM: TileMatmulFamily, VM: TileMatmulFamily, RF: ReaderFamily> StageAttentionFamily
    for DummyStageAttentionFamily<SM, VM, RF>
{
    type Attention<AP: AttentionPrecision> = DummyStageAttention<
        AP,
        SM::Matmul<AP::ES, AP::ES, AP::EA>,
        VM::Matmul<AP::EA, AP::ES, AP::EA>,
        RF::Reader<AP::ES, AttentionTilingLayout>,
        Self::Config,
    >;

    type KeyReader = RF;
    type ValueReader = RF;
    type Config = DummyStageConfig<SM::Config, VM::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        _problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let score_tile_config = SM::setup::<AP::ES, AP::ES, AP::EA, R>(
            client,
            score_tile_matmul_setup_info(selection, line_sizes),
        )?;
        let value_tile_config = VM::setup::<AP::EA, AP::ES, AP::EA, R>(
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
