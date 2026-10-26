//! A [`Projection`]'s carriers read in the kernel: the logical extents a physical shape carries,
//! the line offset of one coordinate, and a flat line's physical coordinates back to its logical
//! one.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::{Axis, Coords, Integer, IntegerExpand, IntegerSeq, IntegerSeqExpand, Projection};

/// The logical extent per axis: the product of `projection`'s physical extents carrying it.
#[cube]
pub(crate) fn logical_extent(
    #[comptime] projection: Projection,
    physical_shape: &Coords<u32>,
) -> Coords<u32> {
    let mut bound = Coords::<u32>::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let picks = comptime!(projection.carriers(projection.logical_axes()[i]).to_vec());
        bound.push(physical_shape.product(picks));
    }
    bound
}

/// The line offset of coordinate `at` along `axis` under `projection`: each physical axis carrying
/// `axis` takes its coordinate of `at`, weighed by its stride. Over a storage-tiled axis the offset
/// is not linear in `at`: a coordinate that crosses into the next stored tile jumps to that tile's
/// stride, so a move from one coordinate to another is the difference of their offsets.
#[cube]
pub(crate) fn axis_offset(
    #[comptime] projection: Projection,
    #[comptime] axis: Axis,
    at: u32,
    physical_shape: &Coords<u32>,
    strides: &Coords<u32>,
) -> u32 {
    let carriers = comptime!(projection.carriers(axis));
    let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
    let mut parts = Sequence::<u32>::new();
    #[unroll]
    for k in 0..comptime!(carriers.len()) {
        let pa = comptime!(carriers[k]);
        let (finer, modulo) = comptime!(projection.carried_by(pa, axis));
        let scale = comptime!(projection.scale(pa, axis) as u32);
        let quot = at.divided_by(physical_shape.product(comptime!(finer.to_vec())));
        let coord = match comptime!(modulo) {
            Some(m) => quot.remainder(physical_shape.at(m)),
            None => quot,
        };
        parts.push(
            coord
                .times(comptime!(scale).runtime())
                .times(strides.at(pa)),
        );
    }
    parts.sum(picks)
}

/// The logical coordinate producing `physical` under `projection`, inverting
/// `BufferLayout`'s `to_source_pos`. Requires [`Projection::is_invertible`].
#[cube]
pub(crate) fn logical_of_physical(
    #[comptime] projection: Projection,
    physical: &Coords<u32>,
    physical_shape: &Coords<u32>,
) -> CoordsDyn {
    comptime!(assert!(
        projection.is_invertible(),
        "Projection::logical_of_physical: not invertible (an affine term mixes several logical \
         coordinates into one physical cell)"
    ));
    let mut out = CoordsDyn::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let axis = comptime!(projection.logical_axes()[i]);
        let carriers = comptime!(projection.carriers(axis));
        let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for k in 0..comptime!(carriers.len()) {
            let pa = comptime!(carriers[k]);
            // Each coordinate weighs the block it sits above, the same finer extents that stripped
            // it.
            let (finer, _) = comptime!(projection.carried_by(pa, axis));
            parts.push(
                physical
                    .at(pa)
                    .times(physical_shape.product(comptime!(finer.to_vec()))),
            );
        }
        out.push(parts.sum(picks));
    }
    out
}

/// Coordinate `j` of flat line `x` under `shape`'s row-major suffix strides.
#[cube]
pub(crate) fn line_coord(x: u32, shape: &Coords<u32>, #[comptime] j: usize) -> u32 {
    let plen = shape.len();
    x.divided_by(shape.product(comptime!(((j + 1)..plen).collect::<Vec<_>>())))
        .remainder(shape.at(j))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);

    /// Weighing the physical coordinates back reconstructs the logical one they were taken from.
    #[test]
    fn logical_of_physical_inverts_what_to_source_pos_reads() {
        let shape = [3usize, 8];
        let p = Projection::new(
            &[A],
            &[crate::PhysicalAxisMap::of(A), crate::PhysicalAxisMap::of(A)],
        );
        assert!(p.is_invertible());
        let block = |pa| {
            p.carried_by(pa, A)
                .0
                .iter()
                .map(|&q| shape[q])
                .product::<usize>()
        };

        for coord in [0usize, 1, 7, 8, 9, 19, 23] {
            let physical = |pa| match p.carried_by(pa, A).1 {
                Some(m) => (coord / block(pa)) % shape[m],
                None => coord / block(pa),
            };
            assert_eq!(physical(0) * block(0) + physical(1) * block(1), coord);
        }
    }
}
