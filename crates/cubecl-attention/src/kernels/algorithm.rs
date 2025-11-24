use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::AttentionElems;
use crate::components::tile::TileAttentionFamily;
use crate::components::{
    AttentionLineSizes, AttentionProblem, AttentionSelection, AttentionSetupError,
    AvailableLineSizes, batch::BatchAttentionFamily, global::GlobalAttentionFamily,
    stage::StageAttentionFamily,
};

pub trait Algorithm {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<<Self::BatchAttention as BatchAttentionFamily>::Config, AttentionSetupError> {
        Self::BatchAttention::setup::<R>(client, problem, selection, line_sizes, dtypes)
    }
}
