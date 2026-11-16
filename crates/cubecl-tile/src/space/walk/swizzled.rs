//! Where a swizzled cube level puts a walk's two in-plane positions.

use cubecl::prelude::*;

use crate::{
    Coords, CubeAxis, DimensionDistribution, Integer, IntegerExpand, Level, Space, cube_positions,
};

/// The swizzled cube level's two in-plane positions; zeros where no order is stated.
#[cube]
pub(crate) fn swizzled_positions(
    #[comptime] space: &Space,
    #[comptime] level: &Level,
    instances: &Coords<usize>,
) -> Coords<usize> {
    let mut out = Coords::<usize>::new();
    if level.order().swizzles() {
        let (x_at, y_at) = comptime!(in_plane_dimensions(space, level));
        let (count_x, count_y) = (instances.at(x_at), instances.at(y_at));
        let flat =
            CubeAxis::position(CubeAxis::X).plus(CubeAxis::position(CubeAxis::Y).times(count_x));
        let (x, y) = cube_positions(flat, (count_x, count_y), level.order());
        out.push(x);
        out.push(y);
    } else {
        out.push(0usize);
        out.push(0usize);
    }
    out
}

/// Where this space holds the cube level's two in-plane dimensions, `(x, y)`. Panics unless each
/// owns its whole grid dimension.
pub(crate) fn in_plane_dimensions(space: &Space, level: &Level) -> (usize, usize) {
    let at = |wanted: CubeAxis| {
        let found: Vec<usize> = (0..space.rank())
            .filter(|&p| level.cube_axis(space.dimension_at(p)) == Some(wanted))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "Walk: a {:?} cube level distributes the grid's {wanted:?} dimension to {} of its dimensions, \
             and a swizzle can put back only a dimension that owns a whole dimension",
            level.order(),
            found.len()
        );
        let p = found[0];
        assert_eq!(
            DimensionDistribution::unspanned_weight(level, space, space.dimension_at(p)),
            1,
            "Walk: a {:?} cube level shares the grid's {wanted:?} dimension with a dimension this \
             space does not span, whose coordinate a swizzle has no way to put back",
            level.order()
        );
        p
    };
    (at(CubeAxis::X), at(CubeAxis::Y))
}
