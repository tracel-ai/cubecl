//! The runtime side of a [`Projection`]: [`RuntimeMap`] and the `#[cube]` digit folds.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::{Axis, Coords, Integer, IntegerExpand, IntegerSeq, IntegerSeqExpand, Projection};

/// The runtime half of a [`Projection`]: its dynamic coefficients and divisors, and the
/// window origin's phase under each divisor.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct RuntimeMap {
    /// Dynamic coefficients and divisors, physical axis major, each axis's divisor last.
    pub(crate) coefficients: Coords<u32>,
    /// The window origin's phase within its divisor per physical axis, `0` when integer.
    pub(crate) residues: Coords<u32>,
}

#[cube]
impl RuntimeMap {
    /// The runtime map of an integer mapping: no coefficients, zero phases.
    pub(crate) fn integral(#[comptime] physical_rank: usize) -> RuntimeMap {
        RuntimeMap {
            coefficients: Coords::<u32>::new(),
            residues: Coords::constant(comptime!(vec![0; physical_rank])),
        }
    }

    /// This map copied into mutable registers that survive a sibling slot's refill.
    pub(crate) fn stored(&self) -> RuntimeMap {
        RuntimeMap {
            coefficients: self.coefficients.stored(),
            residues: self.residues.stored(),
        }
    }

    /// Store a source window's complete runtime addressing state in this slot.
    pub(crate) fn store_from(&mut self, src: &RuntimeMap) {
        self.coefficients.store_from(&src.coefficients);
        self.residues.store_from(&src.residues);
    }
}

/// The logical extent per axis, folded from `projection`'s physical shape.
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
/// `axis` takes its digit of `at`, weighed by its stride. Over a storage-tiled axis the offset is
/// not linear in `at`: a coordinate that crosses into the next stored tile jumps to that tile's
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
        let (finer, modulo) = comptime!(projection.digit(pa, axis));
        let scale = comptime!(projection.scale(pa, axis) as u32);
        let quot = at.divided_by(physical_shape.product(comptime!(finer.to_vec())));
        let digit = match comptime!(modulo) {
            Some(m) => quot.remainder(physical_shape.at(m)),
            None => quot,
        };
        parts.push(
            digit
                .times(comptime!(scale).runtime())
                .times(strides.at(pa)),
        );
    }
    parts.sum(picks)
}

/// The logical coordinate producing physical `digits` under `projection`, inverting
/// `BufferLayout`'s `to_source_pos`. Requires [`Projection::is_invertible`].
#[cube]
pub(crate) fn fold_physical(
    #[comptime] projection: Projection,
    digits: &Coords<u32>,
    physical_shape: &Coords<u32>,
) -> CoordsDyn {
    comptime!(assert!(
        projection.is_invertible(),
        "Projection::fold_physical: not invertible (an affine term mixes several logical \
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
            // Each digit weighs the block it sits above, the same finer extents that stripped it.
            let (finer, _) = comptime!(projection.digit(pa, axis));
            parts.push(
                digits
                    .at(pa)
                    .times(physical_shape.product(comptime!(finer.to_vec()))),
            );
        }
        out.push(parts.sum(picks));
    }
    out
}

/// Digit `j` of flat line `x` under `shape`'s row-major suffix strides.
#[cube]
pub(crate) fn line_digit(x: u32, shape: &Coords<u32>, #[comptime] j: usize) -> u32 {
    let plen = shape.len();
    x.divided_by(shape.product(comptime!(((j + 1)..plen).collect::<Vec<_>>())))
        .remainder(shape.at(j))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);

    /// Folding the digits back reconstructs the coordinate they were decomposed from.
    #[test]
    fn fold_physical_digits_invert_to_source_pos_digits() {
        let shape = [3usize, 8];
        let p = Projection::new(
            &[A],
            &[crate::PhysicalAxisMap::of(A), crate::PhysicalAxisMap::of(A)],
        );
        assert!(p.is_invertible());
        let block = |pa| {
            p.digit(pa, A)
                .0
                .iter()
                .map(|&q| shape[q])
                .product::<usize>()
        };

        for coord in [0usize, 1, 7, 8, 9, 19, 23] {
            let digit = |pa| match p.digit(pa, A).1 {
                Some(m) => (coord / block(pa)) % shape[m],
                None => coord / block(pa),
            };
            assert_eq!(digit(0) * block(0) + digit(1) * block(1), coord);
        }
    }
}
