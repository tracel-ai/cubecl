use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError,
    tile::{
        TileAttentionFamily,
        dummy::{AttentionMatmulFamily, DummyTileAttention},
    },
};

pub struct DummyTileAttentionFamily<FM: AttentionMatmulFamily> {
    _phantom: PhantomData<FM>,
}

impl<FM: AttentionMatmulFamily> TileAttentionFamily for DummyTileAttentionFamily<FM> {
    type Attention<AP: AttentionPrecision> = DummyTileAttention<AP, FM::Matmul<AP>>;

    type Config = FM::Config;

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError> {
        FM::setup::<AP, R>(client, problem, selection, line_sizes, num_planes)
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }
}
