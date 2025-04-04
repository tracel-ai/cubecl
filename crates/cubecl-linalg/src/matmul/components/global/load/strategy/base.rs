use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, LoadingValidation, Quantization};
use crate::matmul::components::stage::{Stage, TilingLayout};
use crate::matmul::components::{InputIdent, MatmulPrecision, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
pub trait AsyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<MP: MatmulPrecision, G: global::GlobalConfig, CM: CopyMechanism<MP::ES>>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        mechanism: &CM,
        quantization: CubeOption<Quantization<MP>>,
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
    fn load_buffer<MP: MatmulPrecision, G: global::GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        buffer_index: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[cube]
pub trait SyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<MP: MatmulPrecision, G: global::GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );
}

#[cube]
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<MP: MatmulPrecision, G: global::GlobalConfig, CM: CopyMechanism<MP::ES>>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        mechanism: &CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}
