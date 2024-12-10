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
pub struct ReduceConfig {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub line_mode: LineMode,
    pub line_size: u32,
    pub bound_checks: bool,
}

impl ReduceConfig {
    pub fn generate<R: Runtime>(
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
            .generate_bound_checks(reduce_count, strategy)
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
        self.line_size = match self.line_mode {
            LineMode::Parallel => tensor_line_size_parallel(
                R::supported_line_sizes(),
                input.shape,
                input.strides,
                axis,
            ) as u32,
            LineMode::Perpendicular => tensor_line_size_perpendicular(
                R::supported_line_sizes(),
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
        self.cube_count = CubeCount::new_1d(reduce_count.div_ceil(agent_count_per_cube));
        self
    }

    fn generate_bound_checks(mut self, reduce_count: u32, strategy: &ReduceStrategy) -> Self {
        // When using the shared strategy, we use exactly one cube per reduction. Thus, there is no need for bound checks.
        if strategy.shared {
            return self;
        }

        let reduce_count_lined = match self.line_mode {
            LineMode::Parallel => reduce_count,
            LineMode::Perpendicular => reduce_count / self.line_size,
        };

        let agent_count_per_cube = if strategy.use_planes {
            self.cube_dim.y
        } else {
            self.cube_dim.num_elems()
        };

        self.bound_checks = reduce_count_lined % agent_count_per_cube != 0;
        self
    }
}
