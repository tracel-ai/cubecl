use crate::{self as cubecl};

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
};

use cubecl_runtime::MmaConfig;
use half::f16;

#[cube(launch)]
pub fn kernel_strided(
    lhs: &Array<f16>,
    rhs: &Array<f16>,
    out: &mut Array<f32>,
    #[comptime] stride_lhs: u32,
    #[comptime] stride_rhs: u32,
) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        stride_lhs,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        stride_rhs,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

pub fn test_cmma_strided<R: Runtime>(client: ComputeClient<R::Server>, cube_dimensions: CubeDim) {
    // Lhs (row major) will have strided tiles
    let (m, n, k) = (16, 16, 32);
    let (t_m, t_n, t_k) = (16, 16, 16);
    if !client.properties().features.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16).into(),
        b_type: ElemType::Float(FloatKind::F16).into(),
        cd_type: ElemType::Float(FloatKind::F32).into(),
        m: t_m as u32,
        k: t_k as u32,
        n: t_n as u32,
    }) {
        // We can't execute the test, skip.
        return;
    }

    // Fills left tile while right tile is zero
    let lhs: Vec<f16> = (0..m * k)
        .map(|i| {
            if (i % k) < t_k {
                f16::from_f32((i - (i / k) * t_k) as f32)
            } else {
                f16::from_f32(0.)
            }
        })
        .collect();
    let rhs: Vec<f16> = (0..n * k).map(|i| f16::from_f32((i % 8) as f32)).collect();

    let lhs = client.create(f16::as_bytes(&lhs));
    let rhs = client.create(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * m * n);

    unsafe {
        kernel_strided::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, m * k, 1),
            ArrayArg::from_raw_parts::<f16>(&rhs, k * n, 1),
            ArrayArg::from_raw_parts::<f32>(&out, m * n, 1),
            k as u32,
            n as u32,
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    let expected = [
        504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504.,
        504., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400.,
        1400., 1400., 1400., 1400., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296.,
        2296., 2296., 2296., 2296., 2296., 2296., 2296., 3192., 3192., 3192., 3192., 3192., 3192.,
        3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 4088., 4088., 4088.,
        4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088.,
        4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984.,
        4984., 4984., 4984., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880.,
        5880., 5880., 5880., 5880., 5880., 5880., 6776., 6776., 6776., 6776., 6776., 6776., 6776.,
        6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 7672., 7672., 7672., 7672.,
        7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 8568.,
        8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568.,
        8568., 8568., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464.,
        9464., 9464., 9464., 9464., 9464., 10360., 10360., 10360., 10360., 10360., 10360., 10360.,
        10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 11256., 11256.,
        11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256.,
        11256., 11256., 11256., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152.,
        12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 13048., 13048., 13048.,
        13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048.,
        13048., 13048., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944.,
        13944., 13944., 13944., 13944., 13944., 13944., 13944.,
    ];

    assert_eq!(expected, actual);
}
