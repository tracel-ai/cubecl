use cubecl_core::{
    ir::Elem, prelude::*, tensor_line_size_parallel, tensor_line_size_perpendicular,
};

use crate::ReduceStrategy;

// TODO: Should we allows the user to change that?
const DEFAULT_CUBE_DIM: CubeDim = CubeDim::new_2d(32, 8);

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub line_mode: LineMode,
    pub line_size: u32,
    pub bound_checks: bool,
}

impl ReduceConfig {
    pub fn new(
        cube_count: CubeCount,
        cube_dim: CubeDim,
        line_mode: LineMode,
        line_size: u32,
    ) -> Self {
        Self {
            cube_count,
            cube_dim,
            line_mode,
            line_size,
            bound_checks: false,
        }
    }

    pub fn do_bound_checks_if(&mut self, condition: bool) {
        self.bound_checks = self.bound_checks || condition;
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LineMode {
    Parallel,
    Perpendicular,
}

pub(crate) fn generate_config<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    strategy: &ReduceStrategy,
    elem: &Elem,
) -> ReduceConfig {
    match (strategy.use_planes, strategy.shared) {
        (false, false) => generate_config_unit::<R>(input, output, axis, elem),
        (true, false) => {
            // This assumes that the strategy is already validated. Thus the plane dim is fixed.
            let plane_dim = client.properties().hardware_properties().plane_size_min;
            generate_config_plane::<R>(input, output, axis, plane_dim, elem)
        }
        (false, true) => generate_config_shared::<R>(input, output, axis, elem),
        (true, true) => unimplemented!(),
    }
}

fn generate_config_unit<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    elem: &Elem,
) -> ReduceConfig {
    let stride = input.strides[axis as usize];
    if stride == 1 {
        generate_config_unit_parallel(input, output, axis, elem)
    } else {
        generate_config_unit_perpendicular(input, output, axis, elem)
    }
}

fn generate_config_unit_parallel<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    elem: &Elem,
) -> ReduceConfig {
    let line_mode = LineMode::Parallel;
    let line_size = generate_line_size(input, axis, line_mode, elem);

    let unit_count = output.size() as u32;
    let cube_dim = DEFAULT_CUBE_DIM;
    let cube_count = CubeCount::new_1d(unit_count.div_ceil(cube_dim.num_elems()));

    let mut config = ReduceConfig::new(cube_count, cube_dim, line_mode, line_size);
    config.do_bound_checks_if(unit_count % cube_dim.num_elems() != 0);
    config
}

fn generate_config_unit_perpendicular<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    elem: &Elem,
) -> ReduceConfig {
    let unit_count = output.size() as u32;

    let cube_dim = DEFAULT_CUBE_DIM;
    let cube_count = CubeCount::new_1d(unit_count.div_ceil(cube_dim.num_elems()));

    let line_mode = LineMode::Perpendicular;
    let line_size = generate_line_size(input, axis, line_mode, elem);

    let mut config = ReduceConfig::new(cube_count, cube_dim, line_mode, line_size);
    config.do_bound_checks_if((unit_count / line_size) % cube_dim.num_elems() != 0);
    config
}

fn generate_config_plane<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    plane_dim: u32,
    elem: &Elem,
) -> ReduceConfig {
    let reduce_count = output.size() as u32;
    let plane_count_per_cube = 8;

    let cube_dim = CubeDim::new_2d(plane_dim, plane_count_per_cube);
    let cube_count = CubeCount::new_1d(reduce_count.div_ceil(plane_count_per_cube));

    let line_mode = LineMode::Parallel;
    let line_size = generate_line_size(input, axis, line_mode, elem);

    let mut config = ReduceConfig::new(cube_count, cube_dim, line_mode, line_size);
    config.do_bound_checks_if(reduce_count % plane_count_per_cube != 0);
    config
}

fn generate_config_shared<R: Runtime>(
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: u32,
    elem: &Elem,
) -> ReduceConfig {
    let stride = input.strides[axis as usize];
    let line_mode = if stride == 1 {
        LineMode::Parallel
    } else {
        LineMode::Perpendicular
    };
    let line_size = generate_line_size(input, axis, line_mode, elem);

    let cube_dim = DEFAULT_CUBE_DIM;

    let reduce_count = output.size() as u32;
    let cube_count = match line_mode {
        LineMode::Parallel => CubeCount::new_1d(reduce_count),
        LineMode::Perpendicular => CubeCount::new_1d(reduce_count / line_size),
    };

    ReduceConfig::new(cube_count, cube_dim, line_mode, line_size)
}

fn generate_line_size<R: Runtime>(
    input: &TensorHandleRef<R>,
    axis: u32,
    mode: LineMode,
    elem: &Elem,
) -> u32 {
    let line_size = match mode {
        LineMode::Parallel => tensor_line_size_parallel(
            R::max_line_size_elem(elem),
            input.shape,
            input.strides,
            axis as usize,
        ),
        LineMode::Perpendicular => tensor_line_size_perpendicular(
            R::max_line_size_elem(elem),
            input.shape,
            input.strides,
            axis as usize,
        ),
    };
    line_size as u32
}
