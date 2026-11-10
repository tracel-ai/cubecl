//! A stage that multiplies each vector by a value of its row.

use cubecl::prelude::*;

use super::base::{VectorAt, VectorStage, VectorStageLaunch};
use crate::launch::arg::merged::row_values::RowValues;

/// Multiplies each vector, in `f32`, by its row's value in a [`RowValues`] an earlier launch wrote.
pub struct RowScale;

#[cube]
impl VectorStage for RowScale {
    type Arg<E: Numeric> = &'static [f32];

    fn vector<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        at: &VectorAt,
        value: Vector<E, N>,
    ) -> Vector<E, N> {
        let factor = arg[at.row()];
        Vector::<E, N>::cast_from(Vector::<f32, N>::cast_from(value) * Vector::new(factor))
    }
}

impl VectorStageLaunch for RowScale {
    type Operand = RowValues;

    fn arg<E: Numeric>(operand: RowValues) -> BufferArg {
        operand.arg()
    }
}
