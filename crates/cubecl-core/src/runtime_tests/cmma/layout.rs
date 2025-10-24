use crate::{self as cubecl, cmma::MmaDefinition};

use cubecl::prelude::*;

use cubecl_ir::MatrixIdent;
use cubecl_runtime::MmaConfig;
use half::f16;

#[cube(launch)]
pub fn test_lhs_layout(a: Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        8,
        8,
        8,
        cmma::MatrixLayout::RowMajor,
        &a.to_slice(),
        16,
    );

    // let elems_per_lane = def.elems_per_lane(MatrixIdent::A);
    let elems_per_lane = 2;
    for i in 0..elems_per_lane {
        let out_offset = UNIT_POS_X * elems_per_lane + i;
        let pos = a[i];
        out[out_offset] = f32::cast_from(pos);
    }
}

pub fn test_layout_a<R: Runtime>(client: ComputeClient<R::Server>, cube_dimensions: CubeDim) {
    // if !client.properties().features.mma.contains(&MmaConfig {
    //     a_type: f16::cube_type(),
    //     b_type: f16::cube_type(),
    //     cd_type: f32::cube_type(),
    //     m: 8 as u32,
    //     n: 8 as u32,
    //     k: 8 as u32,
    // }) {
    //     // We can't execute the test, skip.
    //     println!("Skipping test");
    //     return;
    // }

    let lhs: Vec<f16> = (0..64).map(|i| f16::from_f32(i as f32)).collect();
    let lhs = client.create(f16::as_bytes(&lhs));

    let out = client.empty(core::mem::size_of::<f32>() * 64);

    unsafe {
        test_lhs_layout::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, 64, 1),
            ArrayArg::from_raw_parts::<f16>(&out, 64, 1),
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    println!("{:?}", actual);
    assert!(false)
}
