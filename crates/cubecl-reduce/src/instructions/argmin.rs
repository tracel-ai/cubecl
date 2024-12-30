use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{lowest_coordinate_matching, ArgAccumulator, Reduce, ReduceInstruction};

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
pub struct ArgMin;

impl Reduce for ArgMin {
    type Instruction<In: Numeric> = Self;
}

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
impl<In: Numeric> ReduceInstruction<In> for ArgMin {
    type AccumulatorItem = (Line<In>, Line<u32>);
    type SharedAccumulator = ArgAccumulator<In>;

    fn null_input(#[comptime] line_size: u32) -> Line<In> {
        Line::empty(line_size).fill(In::max_value())
    }

    fn null_accumulator(#[comptime] line_size: u32) -> Self::AccumulatorItem {
        (
            Self::null_input(line_size),
            Line::empty(line_size).fill(u32::MAX),
        )
    }

    fn assign_accumulator(destination: &mut Self::AccumulatorItem, source: &Self::AccumulatorItem) {
        destination.0 = source.0;
        destination.1 = source.1;
    }

    fn reduce(
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        let (candidate_item, candidate_coordinate) = if use_planes {
            let candidate_item = plane_min(item);
            let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);
            (candidate_item, candidate_coordinate)
        } else {
            (item, coordinate)
        };
        Self::choose_argmin(
            accumulator.0,
            accumulator.1,
            candidate_item,
            candidate_coordinate,
        )
    }

    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        Self::choose_argmin(lhs.0, lhs.1, rhs.0, rhs.1)
    }

    fn merge_line<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let line_size = accumulator.0.size();
        if comptime!(line_size > 1) {
            let mut min = In::max_value();
            let mut coordinate = 0;
            #[unroll]
            for k in 0..line_size {
                let acc_element = accumulator.0[k];
                let acc_coordinate = accumulator.1[k];
                // TODO replace with select
                if acc_element == min && acc_coordinate < coordinate {
                    coordinate = acc_coordinate;
                } else if acc_element < min {
                    min = acc_element;
                    coordinate = acc_coordinate;
                }
            }
            Out::cast_from(coordinate)
        } else {
            Out::cast_from(accumulator.1)
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator.1)
    }
}
