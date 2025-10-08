use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

use cubecl_matmul::components::{
    MatmulIdent, MatrixPrecision, StageIdent,
    global::GlobalConfig,
    stage::{StageMemoryConfig, StridedStage},
};

use crate::components::stage::reader::BiasTilingLayout;

/// Special reader to broadcast the 1D bias to the 2D accumulator matrix
#[derive(CubeType)]
pub enum BiasGlobalReader<IP: MatrixPrecision> {
    Some {
        view: View<Line<IP::Global>, Coords2d>,
        stage: StridedStage<IP::Stage, BiasTilingLayout>,
    },
    None,
}

/// Type of the stage reader for the bias reader
pub type BiasStage<E> = CubeOption<StridedStage<E, BiasTilingLayout>>;

#[cube]
impl<IP: MatrixPrecision> BiasGlobalReader<IP> {
    /// Reads all bias tiles into the stage. Unlike normal readers, bias only reads a 1D vector along
    /// the `n` dimension.
    pub fn load_stage<G: GlobalConfig>(&mut self, #[comptime] config: G) {
        match self {
            BiasGlobalReader::Some { view, stage } => {
                let line_size = config.global_line_size(MatmulIdent::Out);
                let num_stage_elements = config.tiling_scheme().elements_in_stage_n();

                let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
                let unit_pos = unit_id * line_size;

                let mut slice = stage.as_slice_mut(line_size);

                if unit_pos < num_stage_elements {
                    let read_line = view.read_checked((0, unit_pos));
                    slice[unit_id] = Line::cast_from(read_line);
                }
            }
            BiasGlobalReader::None => {}
        }
    }

    /// Return the stage contained in this global reader. It will use custom tiling with
    /// a stride of `0`.
    pub fn stage(&self) -> BiasStage<IP::Stage> {
        match self {
            BiasGlobalReader::Some { stage, .. } => CubeOption::new_Some(*stage),
            BiasGlobalReader::None => CubeOption::new_None(),
        }
    }
}

#[cube]
impl<IP: MatrixPrecision> BiasGlobalReader<IP> {
    /// Create a new bias reader from the bias tensor and a global offset `n_offset`.
    pub fn new(
        view: CubeOption<View<Line<IP::Global>, Coords2d>>,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        match view {
            CubeOption::Some(view) => {
                let stage = init_stage::<IP::Stage>(config);

                BiasGlobalReader::<IP>::new_Some(view, stage)
            }
            CubeOption::None => BiasGlobalReader::new_None(),
        }
    }
}

/// Create a new 1D bias stage of size `stage_size_n`.
#[cube]
fn init_stage<ES: Numeric>(
    #[comptime] config: StageMemoryConfig,
) -> StridedStage<ES, BiasTilingLayout> {
    let line_size = config.stage_line_size;

    let smem = SharedMemory::new_lined(
        comptime!(config.elements_in_stage_col() / line_size),
        line_size,
    );

    StridedStage::<ES, BiasTilingLayout>::new_with_smem(smem, StageIdent::Acc, config)
}
