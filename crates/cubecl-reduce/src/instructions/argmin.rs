use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{lowest_coordinate_matching, Reduce, ReduceInstruction, ReduceShared};

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
pub struct ArgMin;

#[cube]
impl ArgMin {
    /// Compare two pairs of items and coordinates and return a new pair
    /// where each element in the lines is the minimal item with its coordinate.
    /// In case of equality, the lowest coordinate is selected.
    pub fn choose_argmin<N: Numeric>(
        items0: Line<N>,
        coordinates0: Line<u32>,
        items1: Line<N>,
        coordinates1: Line<u32>,
    ) -> (Line<N>, Line<u32>) {
        let to_keep = select_many(
            items0.equal(items1),
            coordinates0.less_than(coordinates1),
            items0.less_than(items1),
        );
        let items = select_many(to_keep, items0, items1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (items, coordinates)
    }
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for ArgMin {}

#[cube]
impl<In: Numeric> Reduce<In> for ArgMin {
    type Accumulator = (Line<In>, Line<u32>);

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        (
            Line::empty(line_size).fill(In::MAX),
            Line::empty(line_size).fill(0u32),
        )
    }

    fn null_value() -> In {
        In::MAX
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) {
        let (candidate_item, candidate_coordinate) = if use_planes {
            let candidate_item = plane_min(item);
            let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);
            (candidate_item, candidate_coordinate)
        } else {
            (item, coordinate)
        };
        let (new_item, new_coordinate) = Self::choose_argmin(
            accumulator.0,
            accumulator.1,
            candidate_item,
            candidate_coordinate,
        );
        accumulator.0 = new_item;
        accumulator.1 = new_coordinate;
    }

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, _shape_axis_reduce: u32) -> Out {
        let line_size = accumulator.0.size();
        let mut min = In::MAX.runtime();
        let mut coordinate = 0;
        #[unroll]
        for k in 0..line_size {
            let acc_element = accumulator.0[k];
            let acc_coordinate = accumulator.1[k];
            if acc_element == min && acc_coordinate < coordinate {
                coordinate = acc_coordinate;
            } else if acc_element < min {
                min = acc_element;
                coordinate = acc_coordinate;
            }
        }
        Out::cast_from(coordinate)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator.1)
    }
}

#[cube]
impl<In: Numeric> ReduceShared<In> for ArgMin {
    type Accumulator = (SharedMemory<Line<In>>, SharedMemory<Line<u32>>);
    type AccumulatorItem = (Line<In>, Line<u32>);

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        (
            SharedMemory::new_lined(length, line_size),
            SharedMemory::new_lined(length, line_size),
        )
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        accumulator.0[index] = Line::empty(line_size).fill(In::MAX);
        accumulator.1[index] = Line::empty(line_size).fill(0);
    }

    fn null_value() -> In {
        In::MAX
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) {
        let (candidate_item, candidate_coordinate) = if use_planes {
            let candidate_item = plane_min(item);
            let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);
            (candidate_item, candidate_coordinate)
        } else {
            (item, coordinate)
        };
        let (new_item, new_coordinate) = Self::choose_argmin(
            accumulator.0[destination],
            accumulator.1[destination],
            candidate_item,
            candidate_coordinate,
        );
        accumulator.0[destination] = new_item;
        accumulator.1[destination] = new_coordinate;
    }

    fn fuse_accumulator(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let origin_item = accumulator.0[origin];
        let origin_coordinate = accumulator.1[origin];
        let destination_item = accumulator.0[destination];
        let destination_coordinate = accumulator.1[destination];
        let (new_item, new_coordinate) = Self::choose_argmin(
            origin_item,
            origin_coordinate,
            destination_item,
            destination_coordinate,
        );
        accumulator.0[destination] = new_item;
        accumulator.1[destination] = new_coordinate;
    }

    fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem {
        (accumulator.0[0], accumulator.1[0])
    }

    fn merge_line<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let line_size = accumulator.0.size();
        let mut min = In::MAX.runtime();
        let mut coordinate = 0;
        #[unroll]
        for k in 0..line_size {
            let acc_element = accumulator.0[k];
            let acc_coordinate = accumulator.1[k];
            if acc_element == min && acc_coordinate < coordinate {
                coordinate = acc_coordinate;
            } else if acc_element < min {
                min = acc_element;
                coordinate = acc_coordinate;
            }
        }
        Out::cast_from(coordinate)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator.1)
    }
}
