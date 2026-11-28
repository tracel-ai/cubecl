use crate::{Dimension, Space};

/// The smallest region that units can read and write in: the one every operand's
/// [`VectorTile`](crate::layout::VectorTile) divides.
///
/// NVFP4 values read `{k 16, n 2}`, their scales `{k 16, n 16}` and an `f16` activation `{k 8}`
/// have the vector region `{k 16, n 16}`. Reading one value wide divides any region, so an operand
/// read so constrains nothing.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorRegion {
    region: Space,
}

impl VectorRegion {
    /// The vector region of operands whose vector tiles cover `vector_tiles`.
    ///
    /// # Panics
    ///
    /// On a dynamic extent, which has no multiple.
    pub fn new(vector_tiles: impl IntoIterator<Item = Space>) -> Self {
        let vector_tiles: Vec<Space> = vector_tiles.into_iter().collect();
        Self {
            region: Space::new(&[]).counts_whole_for(&vector_tiles),
        }
    }

    /// How far the region reaches along `dimension`: one where nothing reads several of it.
    pub fn along(&self, dimension: Dimension) -> usize {
        self.region.along(dimension)
    }

    /// Whether the region divides `space`, along every dimension it reaches.
    pub fn divides(&self, space: &Space) -> bool {
        self.region.divides(space)
    }

    /// The region as a box of its dimensions.
    pub fn space(&self) -> &Space {
        &self.region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    fn nvfp4() -> [Space; 3] {
        [
            Space::new(&[(K, 8)]),
            Space::new(&[(K, 16), (N, 16)]),
            Space::new(&[(K, 16), (N, 2)]),
        ]
    }

    /// Reads of different shapes have the smallest region each divides.
    #[test]
    fn reads_of_different_shapes_each_divide_their_region() {
        let region = VectorRegion::new(nvfp4());
        assert_eq!((region.along(K), region.along(N)), (16, 16));
    }

    /// Reads that do not nest along a dimension meet at their least common multiple.
    #[test]
    fn reads_that_do_not_nest_meet_at_their_multiple() {
        let region = VectorRegion::new([Space::new(&[(K, 4)]), Space::new(&[(K, 3)])]);
        assert_eq!(region.along(K), 12);
    }

    /// Nothing reads several values of a dimension no operand names.
    #[test]
    fn a_dimension_nothing_reads_several_of_is_one_wide() {
        assert_eq!(VectorRegion::new(nvfp4()).along(Dimension(0)), 1);
    }

    /// A region divides a space whole along every dimension it reaches.
    #[test]
    fn a_region_divides_the_space_it_is_whole_in() {
        let region = VectorRegion::new(nvfp4());
        assert!(region.divides(&Space::new(&[(N, 48), (K, 5120)])));
    }

    /// A region does not divide a space it overhangs along one dimension.
    #[test]
    fn a_region_does_not_divide_a_space_it_overhangs() {
        let region = VectorRegion::new(nvfp4());
        assert!(!region.divides(&Space::new(&[(N, 24), (K, 5120)])));
    }
}
