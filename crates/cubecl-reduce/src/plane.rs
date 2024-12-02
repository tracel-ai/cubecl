use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{ArgMax, ArgMin, Mean, Prod, Sum};

/// An instruction for the [reduce_plane](reduce_plane) algorithm.
#[cube]
pub trait ReducePlaneInstruction<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator.
    /// The implement works on lines. Most likely, the accumulator is Line<T>
    /// for some CubePrimitive type T instead of simply T.
    type Accumulator: CubeType;

    // /// Create an unitialized accumulator containing length item of the given line_size.
    // fn create_accumulator(#[comptime] length: u32, #[comptime] line_size: u32)
    //     -> Self::Accumulator;

    /// Set the null value of the reduction into the accumulator at the given index.
    fn init_accumulator(
        // accumulator: &mut Self::Accumulator,
        // index: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator;

    // Reduce item and coordinate into the accumulator using plane operation.
    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, coordinate: u32);

    fn null_item(#[comptime] line_size: u32) -> Line<EI>;

    // Write the accumulator into the ouput at the given index.
    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    );
}

#[cube]
pub fn reduce_plane<RD: ReducePlaneInstruction<EI>, EI: Numeric, EO: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<EO>>,
    reduce_dim: u32,
    #[comptime] cube_dim: u32,
    #[comptime] plane_dim: u32,
    #[comptime] exact_shape: bool,
) {
    let line_size = input.line_size();

    // This is expected to be exact, else it is an error made when calling the kernel.
    let planes_per_cube = cube_dim / plane_dim;
    let plane_id_local = UNIT_POS / PLANE_DIM;
    let plane_id_global = CUBE_POS * planes_per_cube + plane_id_local;

    // NOTE MAYBE cube_dim -> planes_per_cube?
    // let mut accumulator = RD::create_accumulator(cube_dim, line_size);
    // RD::init_accumulator(&mut accumulator, UNIT_POS, line_size);
    let mut accumulator = RD::init_accumulator(line_size);

    // Compute the first index where to start the reduction for the current cube.
    // First, compute the coordinate corresponding to the CUBE_POS element of the output tensor
    // Then, use the strides of the input tensor to find the index of the same coordinate
    // in the input tensor.
    let mut offset = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(plane_id_global, axis);
        offset += coordinate * input.stride(axis);
    }

    let num_items_per_unit = if exact_shape {
        input.shape(reduce_dim) / plane_dim
    } else {
        div_ceil(input.shape(reduce_dim), plane_dim)
    };

    for i in 0..num_items_per_unit {
        let coordinate = i * plane_dim + UNIT_POS_PLANE;
        let index = offset + coordinate * input.stride(reduce_dim);
        #[allow(clippy::collapsible_else_if)]
        if exact_shape {
            RD::accumulate(&mut accumulator, input[index], coordinate);
        } else {
            let item = select(
                coordinate < input.shape(reduce_dim),
                input[index],
                RD::null_item(line_size),
            );
            RD::accumulate(&mut accumulator, item, coordinate);
        }
    }

    if UNIT_POS_PLANE == 0 {
        RD::write(
            output,
            &accumulator,
            plane_id_global,
            input.shape(reduce_dim),
        );
    }
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// Implementations for common instructions.

#[cube]
impl<EI: Numeric> ReducePlaneInstruction<EI> for Sum {
    type Accumulator = Line<EI>;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, _coordinate: u32) {
        *accumulator += plane_sum(item);
    }

    fn null_item(#[comptime] line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(*accumulator);
    }
}

#[cube]
impl<EI: Numeric> ReducePlaneInstruction<EI> for Prod {
    type Accumulator = Line<EI>;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(EI::from_int(1))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, _coordinate: u32) {
        *accumulator *= plane_prod(item);
    }

    fn null_item(#[comptime] line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(1))
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(*accumulator);
    }
}

#[cube]
impl<EI: Numeric> ReducePlaneInstruction<EI> for Mean {
    type Accumulator = Line<EI>;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, _coordinate: u32) {
        *accumulator += plane_sum(item);
    }

    fn null_item(#[comptime] line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(
            *accumulator / Line::empty(accumulator.size()).fill(EI::cast_from(shape_reduce_dim)),
        );
    }
}

#[cube]
impl<EI: Numeric> ReducePlaneInstruction<EI> for ArgMax {
    type Accumulator = (Line<EI>, Line<u32>);

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        (
            Line::empty(line_size).fill(EI::MIN),
            Line::empty(line_size).fill(0),
        )
    }

    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, coordinate: u32) {
        let line_size = item.size();

        let (acc_item, acc_coordinate) = accumulator;

        let candidate_item = plane_max(item);

        let is_candidate = item.equal(candidate_item);

        let candidate_coordinate = select_many(
            is_candidate,
            Line::empty(line_size).fill(coordinate),
            Line::empty(line_size).fill(u32::MAX),
        );
        let candidate_coordinate = plane_min(candidate_coordinate);

        let (new_item, new_coordinate) = Self::choose_argmax(
            *acc_item,
            *acc_coordinate,
            candidate_item,
            candidate_coordinate,
        );
        accumulator.0 = new_item;
        accumulator.1 = new_coordinate;
    }

    fn null_item(#[comptime] line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::MIN)
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator.1);
    }
}

#[cube]
impl<EI: Numeric> ReducePlaneInstruction<EI> for ArgMin {
    type Accumulator = (Line<EI>, Line<u32>);

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        (
            Line::empty(line_size).fill(EI::MAX),
            Line::empty(line_size).fill(0),
        )
    }

    fn accumulate(accumulator: &mut Self::Accumulator, item: Line<EI>, coordinate: u32) {
        let (acc_item, acc_coordinate) = accumulator;

        let candidate_item = plane_min(item);

        let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);

        let (new_item, new_coordinate) = Self::choose_argmin(
            *acc_item,
            *acc_coordinate,
            candidate_item,
            candidate_coordinate,
        );
        accumulator.0 = new_item;
        accumulator.1 = new_coordinate;
    }

    fn null_item(#[comptime] line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::MAX)
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator.1);
    }
}

#[cube]
fn lowest_coordinate_matching<E: CubePrimitive>(
    target: Line<E>,
    item: Line<E>,
    coordinate: u32,
) -> Line<u32> {
    let line_size = item.size();
    let is_candidate = item.equal(target);
    let candidate_coordinate = select_many(
        is_candidate,
        Line::empty(line_size).fill(coordinate),
        Line::empty(line_size).fill(u32::MAX),
    );
    plane_min(candidate_coordinate)
}
