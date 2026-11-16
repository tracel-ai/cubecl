//! The relu reading of a score [`Tile`]: each cell weighs `max(score · scale + bias, 0)` on its
//! own, with no running state across blocks, where a softmax ([`OnlineSoftmax`]) weighs a cell
//! against the rest of its slice.
//!
//! [`OnlineSoftmax`]: crate::ops::softmax::OnlineSoftmax

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// Take one block of this score in as relu weights: each cell becomes
    /// `max(score · scale + bias, 0)`, where its holder keeps it. Slices along `axis`, which is the
    /// tile's innermost, as the softmax's are.
    ///
    /// A per-tensor factor the score carries, taken out of an operand it was summed over
    /// ([`Tile::factor`]) and put back on it ([`Tile::scaled`]), is read under it: the factor,
    /// positive as `scale` is, multiplies `scale`. The weights it leaves are whole, so read them
    /// from the tile [`Tile::scaled`] was called on, not from the one it returned.
    ///
    /// `bias` is a procedural tile windowed to the same region as the score. A masked cell's bias
    /// is `E::min_value()`, which weighs it zero; a bias zero everywhere ([`Procedural::zero`]) is
    /// no bias, and each cell is then one multiply and one max. On a plane's window every unit of
    /// the plane calls it, and the caller meets the plane before the weights are read.
    ///
    /// [`Procedural::zero`]: crate::procedural::Procedural::zero
    pub fn relu_weights<E: Float>(&self, bias: &Tile, scale: E, #[comptime] axis: Axis) {
        let scale = scale * E::cast_from(self.factor.whole());
        let slots = self.slices_per_unit(axis);
        let mut none = Array::<E>::new(slots);
        #[unroll]
        for i in 0..slots {
            none[i] = E::from_int(0);
        }
        let mut slices = self.along(axis);
        let unbiased = bias.is_zero();
        match comptime!(unbiased) {
            true => slices.scaled_minus::<E>(scale, &none, CellFunction::Relu),
            false => {
                slices.scale_add::<E>(scale, bias);
                slices.scaled_minus::<E>(E::from_int(1), &none, CellFunction::Relu);
            }
        }
    }
}
