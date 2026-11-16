//! How an operand is read and how it is scaled, as tiles of its own axes ([`OperandReads`]).
//!
//! Two numbers per axis answer what a launch and a contraction ask of a stored tensor: the tile
//! one read of it brings, and the tile one scale covers. A float operand reads one value and
//! carries no scales, so the same statement serves it without asking whether anything is
//! quantized.

use crate::{Axis, Space};

/// The shape of the smallest indivisible tile of something: one value on every axis but the ones
/// it names.
///
/// A read of a `q4` input brings eight values along the axis its words run along and one of every
/// other, so its tile is `{k: 8}`. A scale of that input covers thirty-two along the same axis and
/// one column, so its tile is `{k: 32}`. Nothing here names a matmul axis, so an attention operand
/// states its own the same way.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MicroTile([Option<(Axis, usize)>; MicroTile::MAX_AXES]);

impl MicroTile {
    /// The most axes a tile reaches along: an operand is a matrix, its words running along one
    /// axis and its reads at most taking several of them beside each other along the other.
    const MAX_AXES: usize = 2;

    /// One value: what an unvectorized, unquantized read brings.
    pub fn unit() -> Self {
        MicroTile([None; Self::MAX_AXES])
    }

    /// `extent` values along `axis`, one of every other.
    pub fn along(axis: Axis, extent: usize) -> Self {
        let mut tile = MicroTile::unit();
        tile.0[0] = Some((axis, extent));
        tile
    }

    /// The axes the tile names and its extent along each, in the order it was built.
    pub fn entries(&self) -> impl Iterator<Item = (Axis, usize)> + '_ {
        self.0.iter().flatten().copied()
    }

    /// How far this tile reaches along `axis`: one where it does not name it.
    pub fn extent(&self, axis: Axis) -> usize {
        self.entries()
            .find(|&(a, _)| a == axis)
            .map_or(1, |(_, extent)| extent)
    }

    /// This tile `count` times as far along `axis`: a read of a word along one axis, taken for
    /// several words along it, or for several positions of another at once.
    ///
    /// # Panics
    ///
    /// Where the tile would reach along a third axis.
    pub fn times(mut self, axis: Axis, count: usize) -> Self {
        if let Some((_, extent)) = self.0.iter_mut().flatten().find(|(a, _)| *a == axis) {
            *extent *= count;
        } else if count > 1 {
            let free = self
                .0
                .iter_mut()
                .find(|entry| entry.is_none())
                .expect("a microtile reaches along at most two axes");
            *free = Some((axis, count));
        }
        self
    }

    /// This tile with its axis `from` named `to`: the same reach, over a matrix whose axes another
    /// reader names otherwise.
    pub fn renamed(mut self, from: Axis, to: Axis) -> Self {
        for (axis, _) in self.0.iter_mut().flatten() {
            if *axis == from {
                *axis = to;
            }
        }
        self
    }

    /// The box this tile covers, an axis to each extent.
    pub fn space(&self) -> Space {
        Space::new(&self.entries().collect::<Vec<_>>())
    }

    /// How many values the tile holds.
    pub fn values(&self) -> usize {
        self.entries().map(|(_, extent)| extent).product()
    }

    /// The one axis this tile reaches along, where it reaches past one value.
    pub fn axis(&self) -> Option<Axis> {
        self.entries()
            .find(|&(_, extent)| extent > 1)
            .map(|(a, _)| a)
    }
}

/// How an operand is read and how it is scaled, as tiles of its own axes.
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
    /// What one read of the scales brings, in scales: a word of them along the axis they group
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
    /// The two axes are usually the same one, and the shape where they are not is the one no
    /// scheme writes today.
    pub fn packed(
        lined: Axis,
        read: usize,
        tiled: Axis,
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

    /// These reads with their axis `from` named `to`: the same reads and scales, over a matrix
    /// whose axes another reader names otherwise.
    pub fn renamed(self, from: Axis, to: Axis) -> Self {
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
        let mut extents: Vec<(Axis, usize)> = Vec::new();
        for (axis, _) in self.scale_read.entries().chain(covered.entries()) {
            if extents.iter().all(|&(a, _)| a != axis) {
                extents.push((axis, self.scale_read.extent(axis) * covered.extent(axis)));
            }
        }
        Some(Space::new(&extents))
    }

    /// The axis this operand's scales are grouped along, where they group one.
    pub fn scaled_along(&self) -> Option<Axis> {
        self.scale.as_ref().and_then(|tile| tile.axis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

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
