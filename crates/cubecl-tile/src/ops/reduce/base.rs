//! `c.reduce_axis(input, monoid)` and its accumulating twin at a final tile.

use cubecl::prelude::*;

use super::leaf as reduce;
use crate::*;

#[cube]
impl Tile {
    /// `c = fold(input)`: reduce `input` into `self` across the contracted axes.
    pub fn reduce_axis(&mut self, input: &Tile, #[comptime] monoid: Monoid) {
        let define!(Served) = self.element;
        self.init_identity::<Served>(monoid);
        self.reduce_axis_accumulate(input, monoid);
    }

    /// `c = fold(c, input)`: fold `input` into what `self` holds, which must be seeded.
    pub fn reduce_axis_accumulate(&mut self, input: &Tile, #[comptime] monoid: Monoid) {
        let define!(Served) = self.element;
        let define!(Source) = input.read_element();
        reduce_leaf::<Served, Source>(self, input, monoid)
    }
}

/// Dispatches to the register nest by the accumulator's form.
#[cube]
pub(crate) fn reduce_leaf<Acc: Numeric, In: Numeric>(
    acc: &mut Tile,
    input: &Tile,
    #[comptime] monoid: Monoid,
) {
    let input_space = comptime!(input.place.space.clone());
    let vector_size = input.vector_size();
    comptime!(assert!(
        input_space
            .extent_at(input_space.rank() - 1)
            .is_multiple_of(vector_size),
        "reduce: the input's innermost extent must be divisible by its vector size"
    ));

    let space = comptime!(acc.place.space.clone());
    match &mut acc.kind {
        TileKind::Memory(g) => {
            reduce::memory::<Acc, In>(g, input, space, monoid);
        }
        TileKind::PlaneTile(t) => {
            reduce_plane_tile::<Acc, In>(t, input, space, monoid);
        }
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "reduce_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            reduce_plane_tile::<Acc, In>(&mut t, input, space, monoid);
        }
        TileKind::TmaGmem(_) => panic!("reduce: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) | TileKind::Lines(_) => {
            panic!("reduce: a procedural tile and the plane's units are not an accumulator sink")
        }
    }
}

#[cube]
fn reduce_plane_tile<Acc: Numeric, In: Numeric>(
    tile: &mut PlaneTile,
    input: &Tile,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    match tile {
        PlaneTile::Registers(d) => {
            reduce::register_data::<Acc, In>(d, input, acc_space, monoid);
        }
        PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => {
            panic!(
                "reduce: a hardware mma fragment scatters its rows across units in a \
                 layout the elementwise walk cannot address; reduce into a register, \
                 Gmem or Smem accumulator instead"
            );
        }
    }
}
