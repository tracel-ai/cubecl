use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation,
        load::AsyncFullLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StageConfig, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::AsyncLoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let matrix_layout = config.matrix_layout(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                tiling_dimensions.total_row(),
                tiling_dimensions.total_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                tiling_dimensions.total_col(),
                tiling_dimensions.total_row() / line_size,
            ),
        };
        let unit_count = config.plane_dim() * config.num_planes();

        if unit_count % num_slices != 0 {
            return Err(Box::new(
                "Number of slices must divide number of units evenly",
            ));
        }
        if slice_length % (unit_count / num_slices) != 0 {
            return Err(Box::new(
                "Number of units per slice must divide slice length evenly",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let line_size = config.to_smm_config().stage_line_size(input_ident.into());

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                tiling_dimensions.total_row(),
                tiling_dimensions.total_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                tiling_dimensions.total_col(),
                tiling_dimensions.total_row() / line_size,
            ),
        };

        let unit_count = config.plane_dim() * config.num_planes();

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        Job {
            nth_slice,
            nth_segment,
            segment_length,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for Job {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        _task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::SmmConfig>(
                stage,
                this.nth_slice,
                comptime!(this.input_ident.as_ident()),
                config.to_smm_config(),
            );

        let window: Window<MP::EI> =
            tensor_reader.load_window_in_stage::<G>(this.nth_slice, this.input_ident, config);
        let seg_start = Min::min(this.nth_segment * this.segment_length, window.size);
        let seg_end = Min::min((this.nth_segment + 1) * this.segment_length, window.size);

        let src_segment = window.slice.slice(seg_start, seg_end);
        let mut dest_segment = destination.slice_mut(seg_start, seg_end);

        CM::memcpy_async(
            mechanism,
            &src_segment.try_cast_unchecked(),
            &mut dest_segment,
        );
    }

    fn task_count(_this: &Self) -> comptime_type!(u32) {
        1
    }
}

const GLOBAL_LINE_SIZE: u32 = 8;
const ELEM_SIZE: u32 = 2; // f16: two bytes
const BANK_SIZE: u32 = 4; // 1 bank: 4 bytes
const NUM_BANKS_PER_UNIT: u32 = GLOBAL_LINE_SIZE * ELEM_SIZE / BANK_SIZE; // 4 = GLOBAL_LINE_SIZE / STAGE_LINE_SIZE
const STAGE_LINE_SIZE: u32 = BANK_SIZE / ELEM_SIZE; // 2
const PLANE_DIM: u32 = 32;

/// This algorithm assumes PLANE_DIM = NUM_BANKS (typically 32)
/// If there are more banks than units, it will not use all banks, but it will still be conflict-free
/// If there are fewer banks than units, there will be bank conflicts, but still fewer than with naive access patterns
#[cube]
fn write_smem_swizzled<EI: Numeric>(
    // Vectorized STAGE_LINE_SIZE
    slice_write: &mut SliceMut<Line<EI>>,
    base_address: u32,
    // Vectorized GLOBAL_LINE_SIZE
    data_global: Line<EI>,
) {
    // COMPTIME
    let units_per_group = PLANE_DIM / NUM_BANKS_PER_UNIT; // 8

    // RUNTIME
    let group = UNIT_POS_X / units_per_group; // Unit's group index = 0..4

    // 0..4
    #[unroll]
    for i in 0..NUM_BANKS_PER_UNIT {
        // (0..4 + 0..4) % 4
        // Group 0: 0,1,2,3
        // Group 1: 1,2,3,0
        // Group 2: 2,3,0,1
        // Group 3: 3,0,1,2
        let swizzled_i = (group + i) % NUM_BANKS_PER_UNIT;
        let address = base_address + swizzled_i;

        // in vec STAGE_LINE_SIZE
        // TODO
        let mut data_stage = Line::<EI>::empty(STAGE_LINE_SIZE);
        #[unroll]
        for j in 0..STAGE_LINE_SIZE {
            data_stage[j] = data_global[swizzled_i * STAGE_LINE_SIZE + j];
        }
        slice_write[address] = data_stage;
    }
}
