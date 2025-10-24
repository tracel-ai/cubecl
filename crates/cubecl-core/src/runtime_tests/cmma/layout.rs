use crate::{self as cubecl};

use cubecl::prelude::*;

use half::f16;

#[cube(launch)]
pub fn test_lhs_layout(original_data: &Array<Line<f16>>, out: &mut Array<Line<f32>>) {
    let a = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<Line<f32>>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        Line::cast_from(0.0),
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}
