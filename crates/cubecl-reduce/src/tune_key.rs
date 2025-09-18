use cubecl_core as cubecl;

use cubecl_core::{AutotuneKey, ir::ElemType};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    elem_input: ElemType,
    elem_output: ElemType,
    elem_acc: ElemType,
    potential_line_size: u8,
    axis_is_contiguous: bool,
    #[autotune(anchor(exp(min = 16, max = 4096)))]
    reduce_axis_shape: usize,
    #[autotune(anchor(exp(max = 16384, base = 4)))]
    reduce_count: usize,
}

impl ReduceAutotuneKey {
    pub fn generate(
        elem_input: ElemType,
        elem_output: ElemType,
        elem_acc: ElemType,
        input_shape: &[usize],
        axis_is_contiguous: bool,
        axis: usize,
    ) -> Self {
        let rank = input_shape.len();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let reduce_axis_shape = input_shape[axis];

        let reduce_count = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, shape)| (i != axis).then_some(shape))
            .product();

        let potential_line_size = Self::potential_line_size(elem_input.size(), reduce_axis_shape);

        ReduceAutotuneKey::new(
            elem_input,
            elem_output,
            elem_acc,
            potential_line_size,
            axis_is_contiguous,
            reduce_axis_shape,
            reduce_count,
        )
    }

    fn potential_line_size(elem_size: usize, mut shape: usize) -> u8 {
        let mut potential_line_size = 1;
        let max_bytes_in_line = 16; // 128 bits
        //
        while shape.is_multiple_of(2)
            && potential_line_size as usize * elem_size < max_bytes_in_line
        {
            potential_line_size *= 2;
            shape /= 2;
        }
        potential_line_size
    }
}
