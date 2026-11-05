//! A [`Projection`]'s carriers read in the kernel: the logical extents a physical shape carries,
//! and the line offset of one coordinate.

use cubecl::prelude::*;

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
