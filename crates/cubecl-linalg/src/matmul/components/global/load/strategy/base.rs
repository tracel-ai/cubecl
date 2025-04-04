use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, LoadingValidation};
use crate::matmul::components::stage::{Stage, TilingLayout};
use crate::matmul::components::{InputIdent, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;

#[cube]
pub trait AsyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<EI: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[cube]
pub trait SyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[cube]
pub trait SyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[cube]
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<EI: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}
