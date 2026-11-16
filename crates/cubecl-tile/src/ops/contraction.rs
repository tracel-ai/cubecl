//! Where a contraction reads its operands: each operand's coordinate from the output's and the
//! reduced dimensions', which matmul, `product` and `reduce` all resolve the same way.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// What the output's coordinates handed to [`resolve_nd_coords`] count along the innermost
/// dimension.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AccCoords {
    /// Cells, divided by the operand's width there.
    Cells,
    /// Lines already, read as they are.
    Lines,
}

/// The coordinate `operand` is read at, one entry per dimension of its space: from `acc_coords`
/// (indexed by `acc.position(dimension)`), from `reduce_coords`, or zero for a routed dimension.
/// The innermost dimension is divided by `width`, except an acc coordinate already counted in lines
/// ([`AccCoords::Lines`]).
#[cube]
// `#[cube]` converts each comptime argument with `From`, which gives no `&[Dimension]` from a
// `&Vec`.
#[allow(clippy::ptr_arg)]
pub(crate) fn resolve_nd_coords(
    #[comptime] operand: &Space,
    #[comptime] acc: &Space,
    #[comptime] reduce: &Vec<Dimension>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
    #[comptime] acc_counts: AccCoords,
) -> CoordsDyn {
    let operand_rank = operand.rank();
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..operand_rank {
        let dimension = operand.dimension_at(p);
        let in_acc = acc.contains(dimension);
        let raw_coord = if in_acc {
            let pos = acc.position(dimension);
            acc_coords.at(comptime!(pos))
        } else {
            match comptime!(reduce.iter().position(|&r| r == dimension)) {
                Some(pos) => reduce_coords.at(comptime!(pos)),
                None => 0u32,
            }
        };
        let divides = comptime!(
            p == operand_rank - 1 && width > 1 && (acc_counts == AccCoords::Cells || !in_acc)
        );
        let coord = if divides {
            raw_coord.divided_by(comptime!(width as u32))
        } else {
            raw_coord
        };
        out.push(coord);
    }

    out
}
