use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

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
pub fn write_smem_swizzled<EI: Numeric>(
    // Vectorized STAGE_LINE_SIZE
    slice_write: &mut SliceMut<Line<EI>>,
    base_address: u32,
    // Vectorized GLOBAL_LINE_SIZE
    line_globalsized: Line<EI>,
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
        let mut line_stagesized = Line::<EI>::empty(STAGE_LINE_SIZE);
        #[unroll]
        for j in 0..STAGE_LINE_SIZE {
            line_stagesized[j] = line_globalsized[swizzled_i * STAGE_LINE_SIZE + j];
        }
        slice_write[address] = line_stagesized;
    }
}
