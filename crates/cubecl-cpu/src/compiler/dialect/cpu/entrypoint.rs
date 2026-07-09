use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
fn cpu_entry_point(
    cube_dim_x: u32,
    cube_dim_y: u32,
    cube_dim_z: u32,
    cube_count_x: u32,
    cube_count_y: u32,
    cube_count_z: u32,
    unit_pos_x: u32,
    unit_pos_y: u32,
    unit_pos_z: u32,
) {
    let cube_count_dim_x = cube_count_x * cube_dim_x;
    let cube_count_dim_y = cube_count_y * cube_dim_y;

    let cube_count_dim_xy = cube_count_dim_x * cube_count_dim_y;
    let cube_count_xy = cube_count_x * cube_count_y;
    for cube_pos_z in 0..cube_count_z {
        let absolute_pos_z = cube_pos_z * cube_dim_z + unit_pos_z;
        let absolute_pos_z_corrected = absolute_pos_z * cube_count_dim_xy;
        let cube_pos_z_corrected = cube_pos_z * cube_count_xy;

        for cube_pos_y in 0..cube_count_y {
            let absolute_pos_y = cube_pos_y * cube_dim_y + unit_pos_y;
            let absolute_pos_y_corrected = absolute_pos_y * cube_count_dim_x;
            let absolute_pos_xy_corrected = absolute_pos_z_corrected + absolute_pos_y_corrected;
            let cube_pos_y_corrected = cube_pos_y * cube_count_x;
            let cube_pos_yz_corrected = cube_pos_z_corrected + cube_pos_y_corrected;

            for cube_pos_x in 0..cube_count_x {
                let absolute_pos_x = cube_pos_x * cube_dim_x + unit_pos_x;
                let _absolute_pos = absolute_pos_xy_corrected + absolute_pos_x;
                let _cube_pos = cube_pos_yz_corrected + cube_pos_x;
            }
        }
    }
}

#[pliron_op(
    name = "cpu.entrypoint",
    format = "`entrypoint ` region($0)",
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    verifier = "succ"
)]
pub struct EntrypointOp;
