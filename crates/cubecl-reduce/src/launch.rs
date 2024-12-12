use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{
    fuse_accumulator_inplace, reduce_inplace, reduce_shared_inplace, LineMode, Reduce,
    ReduceConfig, ReduceStrategy, SharedAccumulator,
};

/// Entry point for reduce.
#[allow(clippy::too_many_arguments)]
pub fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: Reduce<In>>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    let settings = ReduceParams {
        shared: strategy.shared.then(|| {
            if strategy.use_planes {
                config.cube_dim.y
            } else {
                config.cube_dim.num_elems()
            }
        }),
        use_planes: strategy.use_planes,
        line_size: config.line_size,
        line_mode: config.line_mode,
        bound_checks: config.bound_checks,
    };
    unsafe {
        reduce_kernel::launch_unchecked::<In, Out, Rd, Run>(
            client,
            config.cube_count,
            config.cube_dim,
            input.as_tensor_arg(config.line_size as u8),
            output.as_tensor_arg(1),
            ScalarArg::new(axis),
            settings,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceParams {
    pub shared: Option<u32>, // shared if Some(x) where x is the accumulator size.
    pub use_planes: bool,
    pub line_size: u32,
    pub line_mode: LineMode,
    pub bound_checks: bool,
}

#[cube(launch_unchecked)]
fn reduce_kernel<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
) {
    let reduce_index = get_reduce_index(params);

    if params.bound_checks && reduce_index >= get_reduce_count(output.len(), params) {
        return;
    }

    let range = ReduceRange::new::<In, Out>(reduce_index, input, output, axis_reduce, params);

    let accumulator = match comptime!((params.shared, params.use_planes)) {
        (Some(accumulator_size), use_planes) => {
            let mut accumulator = reduce_slice_shared::<In, R>(
                input.to_slice(),
                range,
                accumulator_size,
                params.line_size,
                params.line_mode,
                use_planes,
            );
            sync_units();
            reduce_tree::<In, R>(&mut accumulator, accumulator_size)
        }
        (None, true) => {
            reduce_slice_plane::<In, R>(input.to_slice(), range, params.line_size, params.line_mode)
        }
        (None, false) => {
            reduce_slice::<In, R>(input.to_slice(), range, params.line_size, params.line_mode)
        }
    };

    if elected_writer(params) {
        write_to_output::<In, Out, R>(
            output,
            accumulator,
            reduce_index,
            input.shape(axis_reduce),
            params,
        );
    }
}

#[cube]
fn get_reduce_index(#[comptime] params: ReduceParams) -> u32 {
    if params.shared.is_some() {
        CUBE_POS
    } else if params.use_planes {
        CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y
    } else {
        ABSOLUTE_POS
    }
}

#[cube]
fn get_reduce_count(output_size: u32, #[comptime] params: ReduceParams) -> u32 {
    match comptime!(params.line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / params.line_size,
    }
}

#[derive(CubeType)]
pub struct ReduceRange {
    pub start: u32,
    pub end: u32,
    pub step: u32,
}

#[cube]
impl ReduceRange {
    pub fn new<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &Tensor<Line<In>>,
        output: &mut Tensor<Out>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        match comptime!(params.line_mode) {
            LineMode::Parallel => {
                Self::new_parallel::<In, Out>(reduce_index, input, output, axis_reduce, params)
            }
            LineMode::Perpendicular => {
                Self::new_perpendicular::<In, Out>(reduce_index, input, output, axis_reduce, params)
            }
        }
    }

    fn new_parallel<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &Tensor<Line<In>>,
        output: &mut Tensor<Out>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let line_size = params.line_size.runtime();

        let mut start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            start += coordinate * input.stride(axis);
        }
        start /= line_size;

        let end = start + input.shape(axis_reduce) / line_size;
        let end = select(end < input.len(), end, input.len());

        ReduceRange {
            start,
            end,
            step: 1,
        }
    }

    fn new_perpendicular<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &Tensor<Line<In>>,
        output: &mut Tensor<Out>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let line_size = params.line_size.runtime();

        let mut start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index * line_size, axis);
            start += coordinate * input.stride(axis);
        }
        start /= line_size;

        let step = input.stride(axis_reduce) / line_size;

        let end = start + input.shape(axis_reduce) * step;
        let end = select(end < input.len(), end, input.len());

        ReduceRange { start, end, step }
    }
}

#[cube]
fn elected_writer(#[comptime] settings: ReduceParams) -> bool {
    if settings.shared.is_some() {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true
    }
}

#[cube]
fn write_to_output<In: Numeric, Out: Numeric, R: Reduce<In>>(
    output: &mut Tensor<Out>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            output[reduce_index] = R::merge_line::<Out>(accumulator, shape_axis_reduce)
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(accumulator, shape_axis_reduce);

            #[unroll]
            for k in 0..settings.line_size {
                output[settings.line_size * reduce_index + k] = out[k];
            }
        }
    }
}

#[cube]
pub fn reduce_slice<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut index = range.start;
    let mut coordinate = 0;
    while index < range.end {
        let coordinates = match comptime!(line_mode) {
            LineMode::Parallel => {
                let mut coordinates = Line::empty(line_size).fill(coordinate * line_size);
                if line_size > 1 {
                    #[unroll]
                    for j in 0..line_size {
                        coordinates[j] += j;
                    }
                }
                coordinates
            }
            LineMode::Perpendicular => Line::empty(line_size).fill(coordinate),
        };
        reduce_inplace::<N, R>(&mut accumulator, items[index], coordinates, false);
        index += range.step;
        coordinate += 1;
    }
    accumulator
}

#[cube]
pub fn reduce_slice_plane<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut first_index = range.start;
    let mut first_coordinate = 0;
    while first_index < range.end {
        let coordinates = match comptime!(line_mode) {
            LineMode::Parallel => {
                let mut coordinates =
                    Line::empty(line_size).fill((first_coordinate + UNIT_POS_X) * line_size);
                if line_size > 1 {
                    #[unroll]
                    for j in 0..line_size {
                        coordinates[j] += j;
                    }
                }
                coordinates
            }
            LineMode::Perpendicular => Line::empty(line_size).fill(first_coordinate + UNIT_POS_X),
        };

        let index = first_index + UNIT_POS_X * range.step;
        let item = select(index < range.end, items[index], R::null_input(line_size));

        reduce_inplace::<N, R>(&mut accumulator, item, coordinates, true);

        let plane_dim = CUBE_DIM_X;
        first_index += plane_dim * range.step;
        first_coordinate += plane_dim;
    }
    accumulator
}

#[cube]
pub fn reduce_slice_shared<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    range: ReduceRange,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] use_planes: bool,
) -> R::SharedAccumulator {
    // The index used to read and write into the accumulator.
    let accumulator_index = if use_planes { UNIT_POS_Y } else { UNIT_POS };

    let mut accumulator = R::SharedAccumulator::allocate(accumulator_size, line_size);

    R::SharedAccumulator::write(
        &mut accumulator,
        accumulator_index,
        R::null_accumulator(line_size),
    );

    let mut first_index = range.start;
    let mut first_coordinate = 0;
    while first_index < range.end {
        let index = first_index + UNIT_POS * range.step;
        let item = select(index < range.end, items[index], R::null_input(line_size));
        let coordinate = fill_coordinate_line(first_coordinate + UNIT_POS, line_size, line_mode);
        let coordinate = select(
            index < range.end,
            coordinate,
            Line::empty(line_size).fill(u32::MAX),
        );
        reduce_shared_inplace::<N, R>(
            &mut accumulator,
            accumulator_index,
            item,
            coordinate,
            use_planes,
        );
        first_index += range.step * CUBE_DIM;
        first_coordinate += CUBE_DIM;
    }
    accumulator
}

#[cube]
fn fill_coordinate_line(
    first: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Line<u32> {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let mut coordinates = Line::empty(line_size).fill(first * line_size);
            if line_size > 1 {
                #[unroll]
                for j in 0..line_size {
                    coordinates[j] += j;
                }
            }
            coordinates
        }
        LineMode::Perpendicular => Line::empty(line_size).fill(first),
    }
}

/// Use all units within a cube to fuse an accumulator inplace like this with some padding if `size` is not a power of 2.
///
/// ```ignored
///
///     0   1   2   3   4   5   6   7
///     |   |   |   |   |   |   |   |
///     +---+   +---+   +---+   +---+
///     |       |       |       |
///     +-------+       +-------+
///     |               |
///     +---------------+
///     |
///     *
///
/// ```
///
/// The outcome is stored in the first element of the accumulator and also returned by this function for convenience.
#[cube]
pub fn reduce_tree<In: Numeric, Inst: Reduce<In>>(
    accumulator: &mut Inst::SharedAccumulator,
    #[comptime] size: u32,
) -> Inst::AccumulatorItem {
    if comptime!(size.is_power_of_two()) {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_active_units {
                fuse_accumulator_inplace::<In, Inst>(accumulator, destination, origin);
            }
            jump *= 2;
            sync_units();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_remaining_items / 2 {
                fuse_accumulator_inplace::<In, Inst>(accumulator, destination, origin);
            }
            num_remaining_items = div_ceil(num_remaining_items, 2);
            jump *= 2;
            sync_units();
        }
    }
    sync_units();
    Inst::SharedAccumulator::read(accumulator, 0)
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
