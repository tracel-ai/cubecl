use cubecl_core as cubecl;

use cubecl_core::{AutotuneKey, ir::Elem};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    elem_input: Elem,
    elem_output: Elem,
    line_mode: LineMode,
    potential_line_size: u8,
    #[autotune(anchor(max = 2048))]
    reduce_axis_shape: usize,
    #[autotune(anchor(max = 1024))]
    reduce_count: usize,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LineMode {
    Parallel,
    Perpendicular,
    Unknown,
}

impl ReduceAutotuneKey {
    pub fn generate(
        elem_input: Elem,
        elem_output: Elem,
        input_shape: &[usize],
        axis_stride: Option<usize>,
        axis: usize,
    ) -> Self {
        let rank = input_shape.len();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let reduce_axis_shape = input_shape[axis];

        let line_mode = match axis_stride {
            Some(1) => LineMode::Parallel,
            Some(n) if n > 1 => LineMode::Perpendicular,
            _ => LineMode::Unknown,
        };

        let reduce_count = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, shape)| (i != axis).then_some(shape))
            .product();

        let potential_line_size = Self::potential_line_size(elem_input.size(), reduce_axis_shape);

        ReduceAutotuneKey::new(
            elem_input,
            elem_output,
            line_mode,
            potential_line_size,
            reduce_axis_shape,
            reduce_count,
        )
    }

    fn potential_line_size(elem_size: usize, mut shape: usize) -> u8 {
        let mut potential_line_size = 1;
        let max_bytes_in_line = 32;
        while shape % 2 == 0 && potential_line_size as usize * elem_size < max_bytes_in_line {
            potential_line_size *= 2;
            shape /= 2;
        }
        potential_line_size
    }
}
