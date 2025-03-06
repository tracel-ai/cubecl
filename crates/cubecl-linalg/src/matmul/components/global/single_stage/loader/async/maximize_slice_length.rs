use crate::matmul::components::{
    global::{
        tensor_view::{TensorReader, Window},
        GlobalConfig, LoadingValidation,
    },
    stage::{StageView, StridedTilingLayout},
    Ident, InvalidConfigError,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingStrategy, CopyMechanism};

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct MaximizeSliceLengthLoading {}

impl LoadingValidation for MaximizeSliceLengthLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        if config.transpose_load(ident) {
            return Err(Box::new(
                "Transpose load is not supported with window loading.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncLoadingStrategy for MaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;

    fn load<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage_view: &mut StageView<ES>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let num_slices = stage_view.num_slices::<G::SmmConfig>(ident, config.to_smm_config());
        let unit_count = config.plane_dim() * config.num_planes();

        let slices_per_unit = (num_slices + unit_count - 1) / unit_count;

        // Typically there will be only 1 slice per unit
        #[unroll(slices_per_unit==1)]
        for nth_slice_local in 0..slices_per_unit {
            let nth_slice = unit_count * nth_slice_local + UNIT_POS;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(num_slices % unit_count == 0) {
                load_nth_slice::<EG, ES, CM, G>(
                    nth_slice, read_view, stage_view, mechanism, ident, config,
                );
            } else {
                if nth_slice < num_slices {
                    load_nth_slice::<EG, ES, CM, G>(
                        nth_slice, read_view, stage_view, mechanism, ident, config,
                    );
                }
            };
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[cube]
fn load_nth_slice<EG: Numeric, ES: Numeric, CM: CopyMechanism<ES>, G: GlobalConfig>(
    nth_slice: u32,
    read_view: &TensorReader<EG>,
    stage_view: &mut StageView<ES>,
    mechanism: &CM,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) {
    let window: Window<EG> = read_view.load_window_no_tile::<G>(nth_slice, ident, config);
    let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
        stage_view,
        nth_slice,
        ident,
        config.to_smm_config(),
    );

    CM::memcpy_async(
        mechanism,
        &window.slice.try_cast_unchecked(),
        &mut destination,
    );
}
