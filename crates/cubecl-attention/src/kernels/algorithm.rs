use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::tile::TileAttentionFamily;
use crate::components::{
    AttentionBlueprint, AttentionProblem, AttentionSetupError, AvailableLineSizes,
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::components::{AttentionElems, AttentionTilingScheme};

pub trait Algorithm {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily;

    type Settings;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBlueprint, AttentionSetupError>;

    fn dtypes<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        blueprint: &AttentionBlueprint,
    ) -> Result<AttentionElems, AttentionSetupError>;
}

#[derive(Debug, Clone, Default)]
pub struct SharedAttentionSettings {
    pub tiling_scheme: Option<AttentionTilingScheme>,
    pub reuse_key_value: bool,
    pub two_rows_in_array_tile: bool,
}
