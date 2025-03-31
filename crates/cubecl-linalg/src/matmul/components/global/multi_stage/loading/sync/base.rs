use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::{Stage, TilingLayout};
use crate::matmul::components::{Ident, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
pub trait SyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<EI: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        buffer_index: u32,
        scaling: CubeOption<ES>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}
