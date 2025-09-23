use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError,
    tile::{
        TileAttentionFamily,
        dummy::{DummyTileAttention, FlashMatmulFamily},
    },
};

pub struct DummyTileAttentionFamily<FM: FlashMatmulFamily> {
    _phantom: PhantomData<FM>,
}

impl<FM: FlashMatmulFamily> TileAttentionFamily for DummyTileAttentionFamily<FM> {
    type Attention<AP: AttentionPrecision> =
        DummyTileAttention<AP::FlashPrecision, FM::Matmul<AP::FlashPrecision>>;

    type Config = FM::Config;

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        FM::setup::<AP, R>(client, problem, selection, line_sizes)
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }
}
