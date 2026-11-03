//! Writing a [`Projection`] one buffer dim at a time.
//!
//! ```text
//! Projection::dims()
//!     .dim(B)                                        // dim 0 = b
//!     .dim(stencil(&[(OH, stride), (RH, dilation)]).pad(pad)) // dim 1 = oh*stride+rh*dilation-pad
//!     .dim(C)                                        // dim 2 = c
//!     .build()
//! ```

use cubecl::zspace::SmallVec;

#[cfg(test)]
use crate::Composition;
use crate::{Axis, PhysicalAxisMap, Projection, Space};

/// A [`Projection`] under construction.
#[derive(Clone, Debug)]
pub struct DimsBuilder {
    physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    spanning: SmallVec<[Axis; Space::MAX_RANK]>,
}

impl DimsBuilder {
    /// No dim stated yet.
    pub(crate) fn new() -> Self {
        DimsBuilder {
            physical: SmallVec::new(),
            spanning: SmallVec::new(),
        }
    }

    /// The next buffer dim, coarsest first: an [`Axis`] or a [`split`].
    pub fn dim(mut self, dim: impl Into<PhysicalAxisMap>) -> Self {
        self.physical.push(dim.into());
        self
    }

    /// An axis this operand is defined over but addressed by no dim.
    pub fn spanning(mut self, axis: Axis) -> Self {
        self.spanning.push(axis);
        self
    }

    /// The projection, its logical axes derived from the dims, the innermost dim's last.
    /// Panics on no dims, or on an axis both [`spanning`](Self::spanning) and addressed.
    pub fn build(self) -> Projection {
        assert!(
            !self.physical.is_empty(),
            "Projection::dims: an operand has at least one dim"
        );
        let (innermost, outer) = self.physical.split_last().expect("checked non-empty");
        let mut axes: SmallVec<[Axis; Space::MAX_RANK]> = SmallVec::new();
        let mention = |axis: Axis, axes: &mut SmallVec<[Axis; Space::MAX_RANK]>| {
            if !axes.contains(&axis) {
                axes.push(axis);
            }
        };
        for map in outer {
            for term in map.terms() {
                mention(term.axis, &mut axes);
            }
        }
        for &axis in &self.spanning {
            assert!(
                !self.physical.iter().any(|map| map.addresses(axis)),
                "Projection::dims: {axis:?} is stated as spanning (addressed by no dim) but a dim \
                 addresses it"
            );
            mention(axis, &mut axes);
        }
        // Last, so the line runs along it; a storage-tiled axis keeps its earlier position.
        for term in innermost.terms() {
            mention(term.axis, &mut axes);
        }
        Projection::new(&axes, &self.physical)
    }
}

/// A dim that *is* one coordinate: `.dim(K)`.
impl From<Axis> for PhysicalAxisMap {
    fn from(axis: Axis) -> Self {
        PhysicalAxisMap::of(axis)
    }
}

/// One dim cut into blocks, stated in extents, coarsest first:
/// `split(&[(KB, blocks), (KI, block)])` is `kb·block + ki`. Panics on no terms or extent 0.
pub fn split(extents: &[(Axis, usize)]) -> PhysicalAxisMap {
    assert!(
        !extents.is_empty(),
        "split: a dim is cut by at least one axis"
    );
    let mut terms: SmallVec<[(Axis, usize); Space::MAX_RANK]> = SmallVec::new();
    let mut coefficient = 1;
    for &(axis, extent) in extents.iter().rev() {
        assert!(extent > 0, "split: {axis:?} has extent 0");
        terms.push((axis, coefficient));
        coefficient *= extent;
    }
    terms.reverse();
    PhysicalAxisMap::disjoint(&terms)
}

/// One dim several axes slide over, in coefficients: `oh·stride + rh·dilation`.
#[cfg(test)]
pub(crate) fn stencil(coefficients: &[(Axis, usize)]) -> StencilDim {
    StencilDim {
        coefficients: SmallVec::from_slice(coefficients),
        pad: 0,
    }
}

/// A [`stencil`] before its padding is stated.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct StencilDim {
    coefficients: SmallVec<[(Axis, usize); Space::MAX_RANK]>,
    pad: usize,
}

#[cfg(test)]
impl StencilDim {
    /// How many cells before the buffer's first the window's origin sits.
    pub(crate) fn pad(mut self, pad: usize) -> Self {
        self.pad = pad;
        self
    }
}

#[cfg(test)]
impl From<StencilDim> for PhysicalAxisMap {
    fn from(window: StencilDim) -> Self {
        let map = PhysicalAxisMap::affine(&window.coefficients).shifted(-(window.pad as isize));
        debug_assert!(
            window.coefficients.len() == 1 || map.composition() == Composition::Overlapping,
            "a window over several axes takes the overlapping reading"
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Offset;

    const B: Axis = Axis(0);
    const H: Axis = Axis(1);
    const G: Axis = Axis(2);
    const T: Axis = Axis(3);
    const D: Axis = Axis(4);
    const K: Axis = Axis(5);
    const N: Axis = Axis(6);
    const KB: Axis = Axis(7);
    const KI: Axis = Axis(8);
    const OH: Axis = Axis(9);
    const RH: Axis = Axis(10);
    const C: Axis = Axis(11);
    const M: Axis = Axis(12);

    /// One dim's index for a coordinate: `Σ axis·coefficient + offset`.
    fn index(p: &Projection, dim: usize, coordinate: &[(Axis, usize)]) -> isize {
        let sum: usize = coordinate
            .iter()
            .map(|&(axis, at)| at * p.scale(dim, axis))
            .sum();
        let offset = match p.offset(dim) {
            Offset::Static(o) => o,
            Offset::Dynamic => unreachable!("nothing here builds a dynamic offset"),
        };
        sum as isize + offset
    }

    #[test]
    fn one_axis_per_dim_is_direct() {
        let p = Projection::dims().dim(K).dim(N).build();
        assert_eq!(p, Projection::direct(&[K, N]));
    }

    /// `kb·32 + ki` as 4 blocks of 32: `k = 100` lands at block 3, position 4.
    #[test]
    fn split_derives_the_coefficients_from_extents() {
        let p = Projection::dims()
            .dim(N)
            .dim(split(&[(KB, 4), (KI, 32)]))
            .build();

        assert_eq!(p.scale(1, KB), 32);
        assert_eq!(p.scale(1, KI), 1);
        assert_eq!(index(&p, 1, &[(KB, 3), (KI, 4)]), 100);
        assert_eq!(p.composition(), Composition::Disjoint);
        assert_eq!(p.logical_axes(), &[N, KB, KI]);
    }

    #[test]
    fn split_composes_more_than_two_axes() {
        let p = Projection::dims()
            .dim(split(&[(B, 2), (H, 3), (D, 5)]))
            .build();
        assert_eq!(p.scale(0, B), 15);
        assert_eq!(p.scale(0, H), 5);
        assert_eq!(p.scale(0, D), 1);
        assert_eq!(index(&p, 0, &[(B, 1), (H, 2), (D, 4)]), 29);
    }

    /// Split-merge partials, group-major: `g·splits + t`.
    #[test]
    fn a_partition_in_the_middle_keeps_the_innermost_last() {
        let (group, splits) = (4, 8);
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(split(&[(G, group), (T, splits)]))
            .dim(D)
            .build();

        assert_eq!(index(&p, 2, &[(G, 2), (T, 5)]), 21);
        assert_eq!(index(&p, 2, &[(G, 2), (T, 6)]), 22);
        assert_eq!(p.logical_axes(), &[B, H, G, T, D]);
    }

    /// `oh·2 + rh·1 − 1`: adjacent output steps overlap, and `oh = 0` reaches into the pad.
    #[test]
    fn stencil_is_an_overlapping_affine_map_with_padding() {
        let p = Projection::dims()
            .dim(B)
            .dim(stencil(&[(OH, 2), (RH, 1)]).pad(1))
            .dim(C)
            .build();

        assert_eq!(index(&p, 1, &[(OH, 3), (RH, 2)]), 7);
        assert_eq!(index(&p, 1, &[(OH, 4), (RH, 0)]), 7);
        assert_eq!(index(&p, 1, &[(OH, 0), (RH, 0)]), -1);
        assert_eq!(p.composition(), Composition::Overlapping);
        assert_eq!(p.logical_axes(), &[B, OH, RH, C]);
    }

    /// A grouped-query cache: `g` is addressed by no dim and stays before the innermost axis.
    #[test]
    fn a_spanning_axis_sits_before_the_innermost() {
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(T)
            .dim(D)
            .spanning(G)
            .build();

        assert!(!p.addresses(G));
        assert_eq!(p.logical_axes(), &[B, H, T, G, D]);
        p.validate(4).unwrap();
    }

    #[test]
    fn a_repeated_axis_is_storage_tiled() {
        let p = Projection::dims()
            .dim(B)
            .dim(M)
            .dim(K)
            .dim(M)
            .dim(K)
            .dim(C)
            .build();

        assert_eq!(p.logical_axes(), &[B, M, K, C]);
        assert_eq!(p.tiling().fragments(4).as_slice(), &[1, 2, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "stated as spanning")]
    fn spanning_an_addressed_axis_is_refused() {
        let _ = Projection::dims().dim(B).dim(D).spanning(D).build();
    }
}
