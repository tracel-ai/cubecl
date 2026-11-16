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
use crate::{Dimension, PhysicalAxisMap, Projection, Space};

/// A [`Projection`] under construction.
#[derive(Clone, Debug)]
pub struct DimsBuilder {
    physical: SmallVec<[PhysicalAxisMap; Space::MAX_RANK]>,
    spanning: SmallVec<[Dimension; Space::MAX_RANK]>,
}

impl DimsBuilder {
    /// No dim stated yet.
    pub(crate) fn new() -> Self {
        DimsBuilder {
            physical: SmallVec::new(),
            spanning: SmallVec::new(),
        }
    }

    /// The next buffer dim, coarsest first: a [`Dimension`] or a [`split`].
    pub fn dim(mut self, dim: impl Into<PhysicalAxisMap>) -> Self {
        self.physical.push(dim.into());
        self
    }

    /// A dimension this operand is defined over but addressed by no dim.
    pub fn spanning(mut self, dimension: Dimension) -> Self {
        self.spanning.push(dimension);
        self
    }

    /// The projection, its dimensions derived from the dims, the innermost dim's last.
    /// Panics on no dims, or on a dimension both [`spanning`](Self::spanning) and addressed.
    pub fn build(self) -> Projection {
        assert!(
            !self.physical.is_empty(),
            "Projection::dims: an operand has at least one dim"
        );
        let (innermost, outer) = self.physical.split_last().expect("checked non-empty");
        let mut dimensions: SmallVec<[Dimension; Space::MAX_RANK]> = SmallVec::new();
        let mention =
            |dimension: Dimension, dimensions: &mut SmallVec<[Dimension; Space::MAX_RANK]>| {
                if !dimensions.contains(&dimension) {
                    dimensions.push(dimension);
                }
            };
        for map in outer {
            for term in map.terms() {
                mention(term.dimension, &mut dimensions);
            }
        }
        for &dimension in &self.spanning {
            assert!(
                !self.physical.iter().any(|map| map.addresses(dimension)),
                "Projection::dims: {dimension:?} is stated as spanning (addressed by no dim) but a dim \
                 addresses it"
            );
            mention(dimension, &mut dimensions);
        }
        // Last, so the line runs along it; a storage-tiled dimension keeps its earlier position.
        for term in innermost.terms() {
            mention(term.dimension, &mut dimensions);
        }
        Projection::new(&dimensions, &self.physical)
    }
}

/// A dim that *is* one coordinate: `.dim(K)`.
impl From<Dimension> for PhysicalAxisMap {
    fn from(dimension: Dimension) -> Self {
        PhysicalAxisMap::of(dimension)
    }
}

/// One dim cut into blocks, stated in extents, coarsest first:
/// `split(&[(KB, blocks), (KI, block)])` is `kb·block + ki`. Panics on no terms or extent 0.
pub fn split(extents: &[(Dimension, usize)]) -> PhysicalAxisMap {
    assert!(
        !extents.is_empty(),
        "split: a dim is cut by at least one dimension"
    );
    let mut terms: SmallVec<[(Dimension, usize); Space::MAX_RANK]> = SmallVec::new();
    let mut coefficient = 1;
    for &(dimension, extent) in extents.iter().rev() {
        assert!(extent > 0, "split: {dimension:?} has extent 0");
        terms.push((dimension, coefficient));
        coefficient *= extent;
    }
    terms.reverse();
    PhysicalAxisMap::disjoint(&terms)
}

/// One dim several dimensions slide over, in coefficients: `oh·stride + rh·dilation`.
#[cfg(test)]
pub(crate) fn stencil(coefficients: &[(Dimension, usize)]) -> StencilDim {
    StencilDim {
        coefficients: SmallVec::from_slice(coefficients),
        pad: 0,
    }
}

/// A [`stencil`] before its padding is stated.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct StencilDim {
    coefficients: SmallVec<[(Dimension, usize); Space::MAX_RANK]>,
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
            "a window over several dimensions takes the overlapping reading"
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Offset;

    const B: Dimension = Dimension(0);
    const H: Dimension = Dimension(1);
    const G: Dimension = Dimension(2);
    const T: Dimension = Dimension(3);
    const D: Dimension = Dimension(4);
    const K: Dimension = Dimension(5);
    const N: Dimension = Dimension(6);
    const KB: Dimension = Dimension(7);
    const KI: Dimension = Dimension(8);
    const OH: Dimension = Dimension(9);
    const RH: Dimension = Dimension(10);
    const C: Dimension = Dimension(11);
    const M: Dimension = Dimension(12);

    /// One dim's index for a coordinate: `Σ dimension·coefficient + offset`.
    fn index(p: &Projection, dim: usize, coordinate: &[(Dimension, usize)]) -> isize {
        let sum: isize = coordinate
            .iter()
            .map(|&(dimension, at)| at as isize * p.scale(dim, dimension))
            .sum();
        let offset = match p.offset(dim) {
            Offset::Static(o) => o,
            Offset::Dynamic => unreachable!("nothing here builds a dynamic offset"),
        };
        sum + offset
    }

    #[test]
    fn one_dimension_per_dim_is_direct() {
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
        assert_eq!(p.dimensions(), &[N, KB, KI]);
    }

    #[test]
    fn split_composes_more_than_two_dimensions() {
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
        assert_eq!(p.dimensions(), &[B, H, G, T, D]);
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
        assert_eq!(p.dimensions(), &[B, OH, RH, C]);
    }

    /// A grouped-query cache: `g` is addressed by no dim and stays before the innermost dimension.
    #[test]
    fn a_spanning_dimension_sits_before_the_innermost() {
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(T)
            .dim(D)
            .spanning(G)
            .build();

        assert!(!p.addresses(G));
        assert_eq!(p.dimensions(), &[B, H, T, G, D]);
        p.validate(4).unwrap();
    }

    #[test]
    fn a_repeated_dimension_is_storage_tiled() {
        let p = Projection::dims()
            .dim(B)
            .dim(M)
            .dim(K)
            .dim(M)
            .dim(K)
            .dim(C)
            .build();

        assert_eq!(p.dimensions(), &[B, M, K, C]);
        assert_eq!(p.tiling().fragments(4).as_slice(), &[1, 2, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "stated as spanning")]
    fn spanning_an_addressed_dimension_is_refused() {
        let _ = Projection::dims().dim(B).dim(D).spanning(D).build();
    }
}
