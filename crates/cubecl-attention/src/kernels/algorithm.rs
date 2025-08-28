use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, batch::BatchAttentionFamily,
    global::GlobalAttentionFamily, stage::StageAttentionFamily, tile::TileAttentionFamily,
};

pub trait Algorithm {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<<Self::BatchAttention as BatchAttentionFamily>::Config, AttentionSetupError> {
        Self::BatchAttention::setup::<AP, R>(client, problem, selection, line_sizes)
    }
}
