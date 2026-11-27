//! `dst.product(&a, &b)`: the elementwise product of two tiles, each broadcasting over the
//! dimensions it omits.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::ops::contraction::{AccCoords, resolve_nd_coords};
use crate::*;

#[cube]
impl Tile {
    /// `dst = a ⊗ b`, elementwise over the destination's box.
    pub fn product(&mut self, a: &Tile, b: &Tile) {
        a.refuse_factor("Tile::product");
        b.refuse_factor("Tile::product");
        let define!(Served) = self.element;
        let define!(Lhs) = a.read_element();
        let define!(Rhs) = b.read_element();
        self.mul_from::<Served, Lhs, Rhs>(a, b)
    }

    /// The transport: every unit of the cube strides the destination's groups.
    fn mul_from<D: Numeric, A: Numeric, B: Numeric>(&mut self, a: &Tile, b: &Tile) {
        let space = comptime!(self.place.space.clone());
        let width = self.vector_size();
        let per_read = b.vector_size();
        let split = comptime!(GroupWalk::of(&space, &b.place.space, width, per_read));
        let size!(W) = width;
        let size!(F) = per_read;
        let a_reader = a.nd_split::<A, W>();
        let b_reader = b.nd_split::<B, F>();

        let extents = Coords::constant(split.groups.clone());
        let total = split.groups.iter().product::<usize>();

        let mut dst = self.nd_mut::<D, W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let group = group_line(&extents.unravel(i.cast::<u32>()), split.clone());
            let (a_base, b_base) = bases(
                &group,
                &space,
                &a.place.space,
                &b.place.space,
                width,
                per_read,
            );

            // The step is a unit of `b`'s read, so its address is the group's.
            let scales = b_reader.view.read(b_reader.map.to_source_pos(b_base));
            // `a`'s address moves with the step: its map is evaluated once here, and each step
            // walks `f` coordinates on from it.
            let walk = comptime!(WalkUnits::along(split.dimension, 1));
            let a_anchor = a_reader.map.anchor(a_base, &walk);

            #[unroll]
            for f in 0..split.per_read {
                let value = a_reader.view.read(a_reader.map.advance(
                    &a_anchor,
                    &Coords::constant(comptime!(vec![f])),
                    &walk,
                ));
                let left = Vector::<D, W>::cast_from(value);
                let right = Vector::<D, W>::cast_from(scales.extract(f));
                dst.write(stepped(&group, split.at, f), left * right);
            }
            i += workers;
        }
    }
}

/// How a product's walk divides: one broadcast-operand read per group, its values stepped inside
/// it.
#[derive(Clone, Debug)]
struct GroupWalk {
    /// The dimension a read's values step: the one the broadcast operand's own lines run along.
    dimension: Dimension,
    /// Its position in the destination's box.
    at: usize,
    /// Values one read of that operand serves.
    per_read: usize,
    /// The destination's box in lines, cut at [`at`](Self::at) into groups of one read.
    groups: Vec<usize>,
}

impl GroupWalk {
    /// How `dst` divides against `b`, read `width` lines wide with `per_read` values a read.
    fn of(dst: &Space, b: &Space, width: usize, per_read: usize) -> Self {
        let rank = dst.rank();
        let dimension = b.dimension_at(b.rank() - 1);
        let at = dst.position(dimension);
        assert!(
            per_read == 1 || at != rank - 1,
            "Tile::product: {per_read} of `b` arrive per read along {dimension:?}, which is also the dimension the \
             destination's own lines run along, so one line would straddle two of them"
        );
        let mut groups = line_extents(dst, width, 0, rank);
        assert!(
            groups[at].is_multiple_of(per_read),
            "Tile::product: {} positions of {dimension:?} do not divide into reads of {per_read}",
            groups[at]
        );
        groups[at] /= per_read;
        GroupWalk {
            dimension,
            at,
            per_read,
            groups,
        }
    }
}

/// A group's line coordinate in the destination's box.
#[cube]
fn group_line(at: &Coords<u32>, #[comptime] split: GroupWalk) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..split.groups.len() {
        let coord = match p == split.at {
            true => at.at(p).times(comptime!(split.per_read as u32)),
            false => at.at(p),
        };
        out.push(coord);
    }
    out
}

/// Each operand's own coordinate for a group; a dimension it does not span drops out.
#[cube]
fn bases(
    group: &CoordsDyn,
    #[comptime] dst: &Space,
    #[comptime] a: &Space,
    #[comptime] b: &Space,
    #[comptime] width: usize,
    #[comptime] per_read: usize,
) -> (CoordsDyn, CoordsDyn) {
    let mut cells = Coords::<u32>::new();
    #[unroll]
    for p in 0..dst.rank() {
        cells.push(group[p].times(comptime!(match p == dst.rank() - 1 {
            true => width as u32,
            false => 1u32,
        })));
    }
    let empty = Coords::<u32>::new();
    let no_reduce = comptime!(Vec::<Dimension>::new());
    (
        resolve_nd_coords(a, dst, &no_reduce, &cells, &empty, width, AccCoords::Cells),
        resolve_nd_coords(
            b,
            dst,
            &no_reduce,
            &cells,
            &empty,
            per_read,
            AccCoords::Cells,
        ),
    )
}

/// `base` with the stepped dimension advanced by `f`: the one component a step moves.
#[cube]
fn stepped(base: &CoordsDyn, #[comptime] at: usize, #[comptime] f: usize) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(base.len()) {
        let coord = match p == at {
            true => base[p] + comptime!(f as u32),
            false => base[p],
        };
        out.push(coord);
    }
    out
}
