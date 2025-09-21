use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords1d, r#virtual::VirtualTensor},
};

use cubecl_matmul::components::{
    InputPrecision, MatmulIdent, StageIdent,
    global::GlobalConfig,
    stage::{FullStageReader, StageConfig, StageMemory},
};

use crate::components::stage::reader::BiasTilingLayout;

/// Special reader to broadcast the 1D bias to the 2D accumulator matrix
#[derive(CubeType)]
pub enum BiasGlobalReader<IP: InputPrecision> {
    Some {
        view: View<Line<IP::Global>, Coords1d>,
        stage: StageMemory<IP::Stage, BiasTilingLayout>,
    },
    None,
}

/// Type of the stage reader for the bias reader
pub type BiasStageReader<E> = CubeOption<FullStageReader<E, BiasTilingLayout>>;

#[cube]
impl<IP: InputPrecision> BiasGlobalReader<IP> {
    /// Loads all bias tiles into the stage. Unlike normal readers, bias only loads a 1D vector along
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
                    let read_line = view.read_checked(unit_pos);
                    slice[unit_id] = Line::cast_from(read_line);
                }
            }
            BiasGlobalReader::None => {}
        }
    }

    /// Create a reader for the stage contained in this global reader. It will use custom tiling with
    /// a stride of `0`.
    pub fn stage_reader(&self) -> BiasStageReader<IP::Stage> {
        match self {
            BiasGlobalReader::Some { stage, .. } => {
                CubeOption::new_Some(FullStageReader::new(*stage, StageIdent::Acc))
            }
            BiasGlobalReader::None => CubeOption::new_None(),
        }
    }
}

#[cube]
impl<IP: InputPrecision> BiasGlobalReader<IP> {
    /// Create a new bias reader from the bias tensor and a global offset `n_offset`.
    pub fn new<G: GlobalConfig>(
        tensor: CubeOption<VirtualTensor<IP::Global>>,
        n_offset: u32,
        slice_size: u32,
        #[comptime] config: G,
    ) -> Self {
        match tensor {
            CubeOption::Some(tensor) => {
                let stage = init_stage::<IP::Stage, G>(config);
                let view = tensor.as_view().slice_unchecked(n_offset, slice_size);

                BiasGlobalReader::<IP>::new_Some(view, stage)
            }
            CubeOption::None => BiasGlobalReader::new_None(),
        }
    }
}

/// Create a new 1D bias stage of size `stage_size_n`.
#[cube]
fn init_stage<ES: Numeric, G: GlobalConfig>(
    #[comptime] config: G,
) -> StageMemory<ES, BiasTilingLayout> {
    let line_size = config.stage_config().stage_line_size(StageIdent::Acc);

    let smem = SharedMemory::new_lined(
        comptime!(config.tiling_scheme().elements_in_stage_n() / line_size),
        line_size,
    );

    StageMemory::<ES, BiasTilingLayout>::new_with_smem(smem, 1u32)
}
