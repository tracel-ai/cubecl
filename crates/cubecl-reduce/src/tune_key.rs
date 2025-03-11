use cubecl_core as cubecl;

use cubecl_core::{ir::Elem, AutotuneKey};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    elem_input: Elem,
    elem_output: Elem,
    #[autotune(anchor)]
    reduce_axis_shape: usize,
    #[autotune(anchor)]
    reduce_axis_stride: usize,
    #[autotune(anchor)]
    outer_axes_product: usize, // The product of the shapes of all axes with greater strides.
}

impl ReduceAutotuneKey {
    pub fn generate_without_strides(
        elem_input: Elem,
        elem_output: Elem,
        input_shape: &[usize],
        axis: usize,
    ) -> Self {
        let rank = input_shape.len();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let reduce_axis_shape = input_shape[axis];
        let reduce_axis_stride = 0;

        let outer_axes_product = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, shape)| (i != axis).then_some(shape))
            .product();

        Self::new(
            elem_input,
            elem_output,
            reduce_axis_shape,
            reduce_axis_stride,
            outer_axes_product,
        )
    }

    pub fn generate(
        elem_input: Elem,
        elem_output: Elem,
        input_shape: &[usize],
        input_strides: &[usize],
        axis: usize,
    ) -> Self {
        let rank = input_shape.len();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let reduce_axis_shape = input_shape[axis];
        let reduce_axis_stride = input_strides[axis];

        let outer_axes_product = input_strides
            .iter()
            .zip(input_shape.iter())
            .filter_map(|(stride, shape)| (*stride > reduce_axis_stride).then_some(shape))
            .product();

        Self::new(
            elem_input,
            elem_output,
            reduce_axis_shape,
            reduce_axis_stride,
            outer_axes_product,
        )
    }
}
