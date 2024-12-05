use cubecl_core::prelude::*;

use crate::ReduceStrategy;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct ReduceConfig {
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
) -> (CubeCount, CubeDim, ReduceConfig) {
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
) -> (CubeCount, CubeDim, ReduceConfig) {
    let stride = input.strides[axis as usize];
    let shape = input.shape[axis as usize];
    let unit_count = output.size() as u32;

    let (line_mode, line_size) = if stride == 1 {
        (LineMode::Parallel, max_line_size_dividing::<R>(shape))
    } else {
        (LineMode::Perpendicular, max_line_size_dividing::<R>(stride))
    };

    let mut config = ReduceConfig::new(line_mode, line_size, false);

    let cube_dim = CubeDim::new_2d(32, 8); // TODO: Should we allows the use to change that?
    let cube_count = CubeCount::new_1d(unit_count.div_ceil(cube_dim.num_elems()));

    match line_mode {
        LineMode::Parallel => config.do_bound_checks_if(unit_count % cube_dim.num_elems() != 0),
        LineMode::Perpendicular => {
            config.do_bound_checks_if((unit_count / line_size) % cube_dim.num_elems() != 0)
        }
    }

    (cube_count, cube_dim, config)
}

impl ReduceConfig {
    pub fn new(line_mode: LineMode, line_size: u32, use_planes: bool) -> Self {
        Self {
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

fn max_line_size_dividing<R: Runtime>(length: usize) -> u32 {
    R::supported_line_sizes()
        .iter()
        .filter(|&&line_size| length % line_size as usize == 0)
        .max()
        .cloned()
        .unwrap_or(1) as u32
}
