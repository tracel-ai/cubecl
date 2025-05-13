use cubecl_core::{
    channel::ComputeChannel, prelude::*, server::ComputeServer, tensor_line_size_parallel,
    tensor_line_size_perpendicular,
};
use cubecl_std::tensor::is_contiguous;

use crate::ReduceStrategy;

// TODO: Should we allows the user to change that?
const DEFAULT_CUBE_DIM: CubeDim = CubeDim::new_2d(32, 8);
const DEFAULT_PLANE_COUNT: u32 = 8;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LineMode {
    Parallel,
    Perpendicular,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// How bound checks is handled for inner reductions.
pub enum BoundChecksInner {
    /// No bound check is necessary.
    None,
    /// Using a mask is enough for bound checks.
    /// This will still read the memory in an out-of-bound location,
    /// but will replace the value by the null value.
    Mask,
    /// Branching is necessary for bound checks.
    ///
    /// Probably the right setting when performing fuse on read.
    Branch,
}

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub line_mode: LineMode,
    pub line_size_input: u32,
    pub line_size_output: u32,
    pub bound_checks: bool,
    pub bound_checks_inner: BoundChecksInner,
}

impl ReduceConfig {
    pub(crate) fn generate<R: Runtime, In: CubePrimitive>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: &TensorHandleRef<R>,
        output: &TensorHandleRef<R>,
        axis: usize,
        strategy: &ReduceStrategy,
    ) -> ReduceConfig {
        let reduce_count = output.size() as u32;
        ReduceConfig::new()
            .generate_line_mode(input, axis)
            .generate_line_size::<R, In>(input, output, axis)
            .generate_cube_dim(client, strategy.use_planes)
            .generate_cube_count::<R>(reduce_count, strategy)
    }

    fn new() -> Self {
        // This is only a dummy configuration to use as a starting point.
        Self {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            line_mode: LineMode::Parallel,
            line_size_input: 1,
            line_size_output: 1,
            bound_checks: true,
            bound_checks_inner: BoundChecksInner::Mask,
        }
    }

    fn generate_line_mode<R: Runtime>(mut self, input: &TensorHandleRef<R>, axis: usize) -> Self {
        let stride = input.strides[axis];
        self.line_mode = if stride == 1 {
            LineMode::Parallel
        } else {
            LineMode::Perpendicular
        };
        self
    }

    fn generate_line_size<R: Runtime, In: CubePrimitive>(
        mut self,
        input: &TensorHandleRef<R>,
        output: &TensorHandleRef<R>,
        axis: usize,
    ) -> Self {
        let elem = In::as_elem_native_unchecked();
        let supported_line_sizes = R::line_size_elem(&elem);
        self.line_size_input = match self.line_mode {
            LineMode::Parallel => {
                tensor_line_size_parallel(supported_line_sizes, input.shape, input.strides, axis)
                    as u32
            }
            LineMode::Perpendicular => {
                // To compute the maximum line size we can used,
                // we first sort both the input and output axes by increasing strides.
                // As example, consider
                //    input shape = [2, 4, 6, 8]
                //    input stride = [1, 16, 64, 2]
                //    output shape = [2, 1, 6, 8]
                //    output stride = [1, 1, 2, 12]
                //    axis = 1
                //
                // then we have
                //    input sorted axis = [0, 3, 1, 2]
                //    output sorted axis = [0, 1, 2, 3]
                //
                // From that point, we look at all the axes before the target axis in the sorted input.
                // That is [0, 3] in the example.
                // In the output, we remove the target axis leading to [0, 2, 3] in the example.
                //
                // In order to use perpendicular line, we are limited by the number of entries that are both
                // contiguous in the input and output. This is obtained by taking the head of each list until they are different.
                // In the above example, only the 0 axis is contiguous in both tensor, but it output sorted axis were [0, 1, 3, 2] instead,
                // both the 0 and 3 axes would be contiguous in the two tensors.
                // The corresponding number of entries is the product of the shape for the contiguous axes.
                // In the example, it is simply 2.
                //
                // This gives us an upper bound on the line size we can used.
                // Then, we use the regular method to find the best line size that match the device capacities.

                let mut input_axis_and_strides =
                    input.strides.iter().enumerate().collect::<Vec<_>>();
                input_axis_and_strides.sort_by_key(|(_, stride)| *stride);
                let input_sorted_axis = input_axis_and_strides
                    .into_iter()
                    .map(|(a, _)| a)
                    .take_while(|a| *a != axis);

                let mut output_axis_and_strides =
                    output.strides.iter().enumerate().collect::<Vec<_>>();
                output_axis_and_strides.sort_by_key(|(_, stride)| *stride);
                let output_sorted_axis = output_axis_and_strides
                    .into_iter()
                    .filter_map(|(a, _)| (a != axis).then_some(a));

                let max_line_size = input_sorted_axis
                    .zip(output_sorted_axis)
                    .filter_map(|(i, o)| (i == o).then_some(output.shape[i]))
                    .product();

                tensor_line_size_perpendicular(
                    supported_line_sizes.filter(|size| {
                        *size as usize <= max_line_size && max_line_size % *size as usize == 0
                    }),
                    input.shape,
                    input.strides,
                    axis,
                ) as u32
            }
        };

        if self.line_size_input > 1 && self.line_mode == LineMode::Perpendicular {
            // TODO that this can be improved
            let rank = output.strides.len();
            let is_contiguous =
                is_contiguous(&output.shape[axis..rank], &output.strides[axis..rank])
                    && output.strides[rank - 1] == 1;
            let shape = output.shape.get(axis + 1).cloned().unwrap_or(1) as u32;

            if is_contiguous && shape % self.line_size_input == 0 {
                self.line_size_output = self.line_size_input;
            }
        }
        self
    }

    pub fn generate_cube_dim<S: ComputeServer, C: ComputeChannel<S>>(
        mut self,
        client: &ComputeClient<S, C>,
        use_planes: bool,
    ) -> Self {
        self.cube_dim = if use_planes {
            let plane_dim = client.properties().hardware.plane_size_min;
            CubeDim::new_2d(plane_dim, DEFAULT_PLANE_COUNT)
        } else {
            DEFAULT_CUBE_DIM
        };
        self
    }

    pub fn generate_cube_count<R: Runtime>(
        mut self,
        reduce_count: u32,
        strategy: &ReduceStrategy,
    ) -> Self {
        let agent_count_per_cube =  // An agent is either a unit, a plane or a whole cube depending on the strategy.
            match strategy {
                ReduceStrategy { shared: true, .. } => 1,
                ReduceStrategy { use_planes: true, .. } => self.cube_dim.y,
                ReduceStrategy { use_planes: false, .. } => self.cube_dim.num_elems(),
            };
        let reduce_count_per_cube = match self.line_mode {
            LineMode::Parallel => agent_count_per_cube,
            LineMode::Perpendicular => agent_count_per_cube * self.line_size_input,
        };

        let cube_count = reduce_count.div_ceil(reduce_count_per_cube);

        self.do_bound_checks_if(reduce_count_per_cube * cube_count > reduce_count);

        // If needed, we decompose the cube count to be within runtime limitation.
        let (max_x, max_y, _) = R::max_cube_count();
        let mut cube_count_x = cube_count;
        let mut cube_count_y = 1;
        let mut cube_count_z = 1;
        while cube_count_x > max_x {
            cube_count_x /= 2;
            cube_count_y *= 2;
        }
        while cube_count_y > max_y {
            cube_count_y /= 2;
            cube_count_z *= 2;
        }
        self.cube_count = CubeCount::new_3d(cube_count_x, cube_count_y, cube_count_z);
        self.do_bound_checks_if(cube_count_x * cube_count_y != cube_count);

        self
    }

    fn do_bound_checks_if(&mut self, condition: bool) {
        self.bound_checks = self.bound_checks || condition;
    }
}
