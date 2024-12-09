use cubecl_core::{prelude::*, tensor_line_size_parallel, tensor_line_size_perpendicular};

use crate::ReduceStrategy;

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub line_mode: LineMode,
    pub line_size: u32,
    pub use_planes: bool,
    pub bound_checks: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LineMode {
    Parallel,
    Perpendicular,
}

pub fn generate_config<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    strategy: &ReduceStrategy,
) -> ReduceConfig {
    let _ = client; // TODO Remove the argument if we don't use it.
    match (strategy.use_planes, strategy.shared) {
        (false, false) => generate_config_unit::<R>(input, output, axis),
        (false, true) => unimplemented!(),
        (true, false) => unimplemented!(),
        (true, true) => unimplemented!(),
    }
}

fn generate_config_unit<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
) -> ReduceConfig {
    let stride = input.strides[axis as usize];
    if stride == 1 {
        generate_config_unit_parallel(input, output, axis)
    } else {
        generate_config_unit_perpendicular(input, output, axis)
    }
}

fn generate_config_unit_parallel<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
) -> ReduceConfig {
    let unit_count = output.size() as u32;

    let cube_dim = CubeDim::new_2d(32, 8); // TODO: Should we allows the user to change that?
    let cube_count = CubeCount::new_1d(unit_count.div_ceil(cube_dim.num_elems()));

    let line_mode = LineMode::Parallel;
    let line_size = tensor_line_size_parallel(
        R::supported_line_sizes(),
        input.shape,
        output.strides,
        axis as usize,
    ) as u32;

    let mut config = ReduceConfig::new(cube_count, cube_dim, line_mode, line_size, false);
    config.do_bound_checks_if(unit_count % cube_dim.num_elems() != 0);
    config
}

fn generate_config_unit_perpendicular<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
) -> ReduceConfig {
    let unit_count = output.size() as u32;

    let cube_dim = CubeDim::new_2d(32, 8); // TODO: Should we allows the user to change that?
    let cube_count = CubeCount::new_1d(unit_count.div_ceil(cube_dim.num_elems()));

    let line_mode = LineMode::Perpendicular;
    let line_size = tensor_line_size_perpendicular(
        R::supported_line_sizes(),
        input.shape,
        output.strides,
        axis as usize,
    ) as u32;

    let mut config = ReduceConfig::new(cube_count, cube_dim, line_mode, line_size, false);
    config.do_bound_checks_if((unit_count / line_size) % cube_dim.num_elems() != 0);
    config
}

impl ReduceConfig {
    pub fn new(
        cube_count: CubeCount,
        cube_dim: CubeDim,
        line_mode: LineMode,
        line_size: u32,
        use_planes: bool,
    ) -> Self {
        Self {
            cube_count,
            cube_dim,
            line_mode,
            line_size,
            use_planes,
            bound_checks: false,
        }
    }

    pub fn do_bound_checks_if(&mut self, condition: bool) {
        self.bound_checks = self.bound_checks || condition;
    }
}
