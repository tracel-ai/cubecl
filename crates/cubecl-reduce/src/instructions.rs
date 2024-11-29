use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
pub struct ReduceArgMax;

#[cube]
impl ReduceArgMax {
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

/// Compute the coordinate of the minimum item returning the smallest coordinate in case of equality.
pub struct ReduceArgMin;

#[cube]
impl ReduceArgMin {
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

pub struct ReduceMean;
pub struct ReduceSum;
pub struct ReduceProd;
