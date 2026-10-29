//! Where a contraction reads its operands: each operand's coordinate from the output's and the
//! reduced axes', which matmul, `product` and `reduce` all resolve the same way.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// What the output's coordinates handed to [`resolve_nd_coords`] count along the innermost axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AccCoords {
    /// Cells, divided by the operand's width there.
    Cells,
    /// Lines already, read as they are.
    Lines,
}

/// The coordinate `operand` is read at, one entry per axis of its space: from `acc_coords`
/// (indexed by `acc.position(axis)`), from `reduce_coords`, or zero for a routed axis.
/// The innermost axis is divided by `width`, except an acc coordinate already counted in lines
/// ([`AccCoords::Lines`]).
#[cube]
// `#[cube]` converts each comptime argument with `From`, which gives no `&[Axis]` from a `&Vec`.
#[allow(clippy::ptr_arg)]
pub(crate) fn resolve_nd_coords(
    #[comptime] operand: &Space,
    #[comptime] acc: &Space,
    #[comptime] reduce: &Vec<Axis>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
    #[comptime] acc_counts: AccCoords,
) -> CoordsDyn {
    let operand_rank = comptime!(operand.rank());
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..operand_rank {
        let axis = comptime!(operand.axis_at(p));
        let in_acc = comptime!(acc.contains(axis));
        let raw_coord = if comptime!(in_acc) {
            let pos = comptime!(acc.position(axis));
            acc_coords.at(comptime!(pos))
        } else {
            match comptime!(reduce.iter().position(|&r| r == axis)) {
                Some(pos) => reduce_coords.at(comptime!(pos)),
                None => 0u32,
            }
        };
        let divides = comptime!(
            p == operand_rank - 1 && width > 1 && (acc_counts == AccCoords::Cells || !in_acc)
        );
        let coord = if comptime!(divides) {
            raw_coord.divided_by(comptime!(width as u32))
        } else {
            raw_coord
        };
        out.push(coord);
    }

    out
}
