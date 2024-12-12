use cubecl_core::{
    channel::ComputeChannel, prelude::*, server::ComputeServer, tensor_line_size_parallel,
    tensor_line_size_perpendicular,
};

use crate::ReduceStrategy;

// TODO: Should we allows the user to change that?
const DEFAULT_CUBE_DIM: CubeDim = CubeDim::new_2d(32, 8);
const DEFAULT_PLANE_COUNT: u32 = 8;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LineMode {
    Parallel,
    Perpendicular,
}

#[derive(Debug, Clone)]
pub(crate) struct ReduceConfig {
    pub(crate) cube_count: CubeCount,
    pub(crate) cube_dim: CubeDim,
    pub(crate) line_mode: LineMode,
    pub(crate) line_size: u32,
    pub(crate) bound_checks: bool,
}

impl ReduceConfig {
    pub(crate) fn generate<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: &TensorHandleRef<R>,
        output: &TensorHandleRef<R>,
        axis: usize,
        strategy: &ReduceStrategy,
    ) -> ReduceConfig {
        let reduce_count = output.size() as u32;
        ReduceConfig::new()
            .generate_line_mode(input, axis)
            .generate_line_size(input, axis)
            .generate_cube_dim(client, strategy.use_planes)
            .generate_cube_count(reduce_count, strategy)
    }

    fn new() -> Self {
        // This is only a dummy configuration to use as a starting point.
        Self {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            line_mode: LineMode::Parallel,
            line_size: 1,
            bound_checks: true,
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

    fn generate_line_size<R: Runtime>(mut self, input: &TensorHandleRef<R>, axis: usize) -> Self {
        let supported_line_sizes = 
                R::supported_line_sizes().iter().cloned();
        self.line_size = match self.line_mode {
            LineMode::Parallel => tensor_line_size_parallel(
                supported_line_sizes,
                input.shape,
                input.strides,
                axis,
            ) as u32,
            LineMode::Perpendicular => tensor_line_size_perpendicular(
                supported_line_sizes,
                input.shape,
                input.strides,
                axis,
            ) as u32,
        };
        self
    }

    fn generate_cube_dim<S: ComputeServer, C: ComputeChannel<S>>(
        mut self,
        client: &ComputeClient<S, C>,
        use_planes: bool,
    ) -> Self {
        self.cube_dim = if use_planes {
            let plane_dim = client.properties().hardware_properties().plane_size_min;
            CubeDim::new_2d(plane_dim, DEFAULT_PLANE_COUNT)
        } else {
            DEFAULT_CUBE_DIM
        };
        self
    }

    fn generate_cube_count(mut self, reduce_count: u32, strategy: &ReduceStrategy) -> Self {
        let agent_count_per_cube =  // An agent is either a unit, a plane or a whole cube depending on the strategy.
            match strategy {
                ReduceStrategy { shared: true, .. } => 1,
                ReduceStrategy { use_planes: true, .. } => self.cube_dim.y,
                ReduceStrategy { use_planes: false, .. } => self.cube_dim.num_elems(),
            };
        let reduce_count_per_cube = match self.line_mode {
            LineMode::Parallel => agent_count_per_cube,
            LineMode::Perpendicular => agent_count_per_cube * self.line_size,
        };

        let cube_count = reduce_count.div_ceil(reduce_count_per_cube);

        self.do_bound_checks_if(reduce_count_per_cube * cube_count > reduce_count);
        self.cube_count = CubeCount::new_1d(cube_count);

        self
    }

    fn do_bound_checks_if(&mut self, condition: bool) {
        self.bound_checks = self.bound_checks || condition;
    }
}
