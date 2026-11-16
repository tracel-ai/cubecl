//! How an operand is read and how it is scaled, as tiles of its own dimensions ([`OperandReads`]).
//!
//! Two numbers per dimension answer what a launch and a contraction ask of a stored tensor: the tile
//! one read of it brings, and the tile one scale covers. A float operand reads one value and
//! carries no scales, so the same statement serves it without asking whether anything is
//! quantized.

use crate::{Dimension, Space};

/// The shape of the smallest indivisible tile of something: one value on every dimension but the ones
/// it names.
///
/// A read of a `q4` input brings eight values along the dimension its words run along and one of every
/// other, so its tile is `{k: 8}`. A scale of that input covers thirty-two along the same dimension and
/// one column, so its tile is `{k: 32}`. Nothing here names a matmul dimension, so an attention operand
/// states its own the same way.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MicroTile([Option<(Dimension, usize)>; MicroTile::MAX_DIMENSIONS]);

impl MicroTile {
    /// The most dimensions a tile reaches along: an operand is a matrix, its words running along one
    /// dimension and its reads at most taking several of them beside each other along the other.
    const MAX_DIMENSIONS: usize = 2;

    /// One value: what an unvectorized, unquantized read brings.
    pub fn unit() -> Self {
        MicroTile([None; Self::MAX_DIMENSIONS])
    }

    /// `extent` values along `dimension`, one of every other.
    pub fn along(dimension: Dimension, extent: usize) -> Self {
        let mut tile = MicroTile::unit();
        tile.0[0] = Some((dimension, extent));
        tile
    }

    /// The dimensions the tile names and its extent along each, in the order it was built.
    pub fn entries(&self) -> impl Iterator<Item = (Dimension, usize)> + '_ {
        self.0.iter().flatten().copied()
    }

    /// How far this tile reaches along `dimension`: one where it does not name it.
    pub fn extent(&self, dimension: Dimension) -> usize {
        self.entries()
            .find(|&(a, _)| a == dimension)
            .map_or(1, |(_, extent)| extent)
    }

    /// This tile `count` times as far along `dimension`: a read of a word along one dimension, taken for
    /// several words along it, or for several positions of another at once.
    ///
    /// # Panics
    ///
    /// Where the tile would reach along a third dimension.
    pub fn times(mut self, dimension: Dimension, count: usize) -> Self {
        if let Some((_, extent)) = self.0.iter_mut().flatten().find(|(a, _)| *a == dimension) {
            *extent *= count;
        } else if count > 1 {
            let free = self
                .0
                .iter_mut()
                .find(|entry| entry.is_none())
                .expect("a microtile reaches along at most two dimensions");
            *free = Some((dimension, count));
        }
        self
    }

    /// This tile with its dimension `from` named `to`: the same reach, over a matrix whose dimensions another
    /// reader names otherwise.
    pub fn renamed(mut self, from: Dimension, to: Dimension) -> Self {
        for (dimension, _) in self.0.iter_mut().flatten() {
            if *dimension == from {
                *dimension = to;
            }
        }
        self
    }

    /// The box this tile covers, a dimension to each extent.
    pub fn space(&self) -> Space {
        Space::new(&self.entries().collect::<Vec<_>>())
    }

    /// How many values the tile holds.
    pub fn values(&self) -> usize {
        self.entries().map(|(_, extent)| extent).product()
    }

    /// The one dimension this tile reaches along, where it reaches past one value.
    pub fn dimension(&self) -> Option<Dimension> {
        self.entries()
            .find(|&(_, extent)| extent > 1)
            .map(|(a, _)| a)
    }
}

/// How an operand is read and how it is scaled, as tiles of its own dimensions.
///
/// Both are the same kind of statement, the smallest piece something is true of, so nothing here
/// records which side of a product is quantized, or which way anything runs. An operand carrying
/// no scales is a tile and nothing else, which is what a float tensor is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OperandReads {
    /// What one read brings.
    pub read: MicroTile,
    /// What one scale covers, where this operand carries scales.
    pub scale: Option<MicroTile>,
    /// What one read of the scales brings, in scales: a word of them along the dimension they group
    /// (as many as share a word, which is what their own field says), and the columns beside it
    /// where the scales are stored so. One scale where this operand carries none.
    pub scale_read: MicroTile,
}

impl OperandReads {
    /// One value a read, no scales: what a float tensor is.
    pub fn plain() -> Self {
        OperandReads {
            read: MicroTile::unit(),
            scale: None,
            scale_read: MicroTile::unit(),
        }
    }

    /// An operand whose read brings `read` values along `lined`, whose scales each cover `scale`
    /// along `tiled`, and whose scales come `scales_per_word` to a word.
    ///
    /// The two dimensions are usually the same one, and the shape where they are not is the one no
    /// scheme writes today.
    pub fn packed(
        lined: Dimension,
        read: usize,
        tiled: Dimension,
        scale: usize,
        scales_per_word: usize,
    ) -> Self {
        OperandReads {
            read: MicroTile::along(lined, read),
            scale: Some(MicroTile::along(tiled, scale)),
            scale_read: MicroTile::along(tiled, scales_per_word),
        }
    }

    /// Whether one read brings one value and no scale: a float tensor.
    pub fn is_plain(&self) -> bool {
        self.read.values() == 1 && self.scale.is_none()
    }

    /// These reads with their dimension `from` named `to`: the same reads and scales, over a matrix
    /// whose dimensions another reader names otherwise.
    pub fn renamed(self, from: Dimension, to: Dimension) -> Self {
        OperandReads {
            read: self.read.renamed(from, to),
            scale: self.scale.map(|scale| scale.renamed(from, to)),
            scale_read: self.scale_read.renamed(from, to),
        }
    }

    /// The box one read of this operand covers.
    pub fn read_space(&self) -> Space {
        self.read.space()
    }

    /// The box one read of this operand's scales covers over its values: the scales a read
    /// brings, each spanning the values it scales. NVFP4's scales in tile order, a word of four
    /// blocks of sixteen down one column, cover `{k 64}`. `None` where it carries no scales.
    pub fn scale_read_space(&self) -> Option<Space> {
        let covered = self.scale.as_ref()?;
        let mut extents: Vec<(Dimension, usize)> = Vec::new();
        for (dimension, _) in self.scale_read.entries().chain(covered.entries()) {
            if extents.iter().all(|&(a, _)| a != dimension) {
                extents.push((
                    dimension,
                    self.scale_read.extent(dimension) * covered.extent(dimension),
                ));
            }
        }
        Some(Space::new(&extents))
    }

    /// The dimension this operand's scales are grouped along, where they group one.
    pub fn scaled_along(&self) -> Option<Dimension> {
        self.scale.as_ref().and_then(|tile| tile.dimension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    /// NVFP4 read along `k`, its scales in tile order two rows side by side: a read of scales is a
    /// word of four blocks of sixteen down each row, `{k 64, n 2}`.
    #[test]
    fn a_read_of_scales_covers_the_blocks_it_brings() {
        let mut reads = OperandReads::packed(K, 8, K, 16, 4);
        reads.scale_read = reads.scale_read.times(N, 2);
        assert_eq!(
            reads.scale_read_space(),
            Some(Space::new(&[(K, 64), (N, 2)]))
        );
        assert_eq!(reads.read_space(), Space::new(&[(K, 8)]));
    }

    /// A float operand is plain; a packed one, or one read several values at a time, is not.
    #[test]
    fn a_float_operand_is_plain() {
        assert!(OperandReads::plain().is_plain());
        assert!(!OperandReads::packed(K, 8, K, 16, 4).is_plain());
        assert_eq!(OperandReads::plain().scale_read_space(), None);
    }
}
