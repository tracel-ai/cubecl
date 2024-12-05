use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{lowest_coordinate_matching, Reduce, ReduceInstruction};

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
pub struct ArgMax;

#[cube]
impl ArgMax {
    /// Compare two pairs of items and coordinates and return a new pair
    /// where each element in the lines is the maximal item with its coordinate.
    /// In case of equality, the lowest coordinate is selected.
    pub fn choose_argmax<N: Numeric>(
        items0: Line<N>,
        coordinates0: Line<u32>,
        items1: Line<N>,
        coordinates1: Line<u32>,
    ) -> (Line<N>, Line<u32>) {
        let to_keep = select_many(
            items0.equal(items1),
            coordinates0.less_than(coordinates1),
            items0.greater_than(items1),
        );
        let items = select_many(to_keep, items0, items1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (items, coordinates)
    }
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for ArgMax {}

#[cube]
impl<In: Numeric> Reduce<In> for ArgMax {
    type Accumulator = (Line<In>, Line<u32>);

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        (
            Line::empty(line_size).fill(In::MIN),
            Line::empty(line_size).fill(0u32),
        )
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) {
        let (candidate_item, candidate_coordinate) = if use_planes {
            let candidate_item = plane_max(item);
            let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);
            (candidate_item, candidate_coordinate)
        } else {
            (item, coordinate)
        };
        let (new_item, new_coordinate) = Self::choose_argmax(
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
        let mut max = In::MIN.runtime();
        let mut coordinate = 0;
        #[unroll]
        for k in 0..line_size {
            if accumulator.0[k] == max && accumulator.1[k] < coordinate {
                coordinate = accumulator.1[k];
            } else if accumulator.0[k] > max {
                max = accumulator.0[k];
                coordinate = accumulator.1[k];
            }
        }
        Out::cast_from(coordinate)
    }

    fn to_output_parallel<Out: Numeric>(
        accumulator: Self::Accumulator,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator.1)
    }
}

// #[cube]
// impl<In: Numeric> ReduceShared<In> for ArgMax {
//     type Accumulator = SharedMemory<Line<In>>;
//     type AccumulatorItem = Line<In>;

//     fn create_accumulator(
//         #[comptime] length: u32,
//         #[comptime] line_size: u32,
//     ) -> Self::Accumulator {
//         SharedMemory::new_lined(length, line_size)
//     }

//     fn init_accumulator(
//         accumulator: &mut Self::Accumulator,
//         index: u32,
//         #[comptime] line_size: u32,
//     ) {
//         accumulator[index] = Line::empty(line_size).fill(In::from_int(0))
//     }

//     fn reduce(
//         accumulator: &mut Self::Accumulator,
//         destination: u32,
//         item: Line<In>,
//         _coordinate: Line<u32>,
//         #[comptime] use_plane: bool,
//     ) {
//         if use_plane {
//             accumulator[destination] += plane_sum(item);
//         } else {
//             accumulator[destination] += item;
//         }
//     }

//     fn fuse_accumulator(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
//         let item = accumulator[origin];
//         accumulator[destination] += item;
//     }

//     fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem {
//         accumulator[0]
//     }

//     fn merge_line<Out: Numeric>(
//         accumulator: Self::AccumulatorItem,
//         _shape_axis_reduce: u32,
//     ) -> Out {
//         let mut prod = In::from_int(0);
//         #[unroll]
//         for k in 0..accumulator.size() {
//             prod *= accumulator[k];
//         }
//         Out::cast_from(prod)
//     }

//     fn to_output_parallel<Out: Numeric>(
//         accumulator: Self::AccumulatorItem,
//         _shape_axis_reduce: u32,
//     ) -> Line<Out> {
//         Line::cast_from(accumulator)
//     }
// }
