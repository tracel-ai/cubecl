//! The order the cube level hands its boxes to the grid ([`CubeOrder`]).

use crate::{AxisDistribution, Coords, CubeAxis, Integer, IntegerExpand, Level, Space};
use cubecl::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

/// The order a cube level distributes its boxes to the grid. A swizzle width need not divide
/// the grid; the last strip is ragged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CubeOrder {
    /// The box at `(x, y)` goes to the cube at `(x, y)`.
    #[default]
    RowMajor,
    /// Cubes snake down the second axis in strips `width` wide along the first.
    SwizzleRow(usize),
    /// Cubes snake along the first axis in strips `width` wide along the second.
    SwizzleCol(usize),
}

impl CubeOrder {
    /// A strip one box wide is [`RowMajor`](CubeOrder::RowMajor).
    pub fn canonicalize(self) -> Self {
        match self {
            CubeOrder::SwizzleRow(1) | CubeOrder::SwizzleCol(1) => CubeOrder::RowMajor,
            other => other,
        }
    }

    /// Whether this order permutes anything.
    pub(crate) fn swizzles(self) -> bool {
        !matches!(self.canonicalize(), CubeOrder::RowMajor)
    }
}

/// The two in-plane positions this order gives the cube at flat dispatch index `flat`.
#[cube]
pub(crate) fn cube_positions(
    flat: usize,
    cubes: (usize, usize),
    #[comptime] order: CubeOrder,
) -> (usize, usize) {
    let (count_x, count_y) = cubes;
    match comptime!(order.canonicalize()) {
        CubeOrder::RowMajor => (flat.remainder(count_x), flat.divided_by(count_x)),
        CubeOrder::SwizzleRow(width) => {
            let (step, along) = swizzle_ragged(flat, count_y, comptime!(width as u32), count_x);
            (along as usize, step as usize)
        }
        CubeOrder::SwizzleCol(width) => {
            let (step, along) = swizzle_ragged(flat, count_x, comptime!(width as u32), count_y);
            (step as usize, along as usize)
        }
    }
}

/// [`swizzle`](fn@swizzle) over strips cut from an axis of `strip_axis` elements that
/// `step_length` need not divide; the last strip holds what is left.
#[cube]
pub(crate) fn swizzle_ragged(
    index: usize,
    num_steps: usize,
    #[comptime] step_length: u32,
    strip_axis: usize,
) -> Coords2d {
    comptime!(assert!(
        step_length > 0,
        "swizzle: a strip holds at least one box"
    ));
    let full = num_steps as u32 * step_length;
    let index = index as u32;
    let strip_index = index / full;
    let strip_offset = step_length * strip_index;
    // An index past the grid leaves nothing, and takes a whole width rather than divide by zero.
    let left = (strip_axis as u32).saturating_sub(strip_offset);
    let width = if left < step_length && left > 0 {
        left
    } else {
        step_length.runtime()
    };
    snake(
        strip_index,
        index - strip_index * full,
        num_steps,
        width,
        strip_offset,
    )
}

#[cube]
/// Maps a linear `index` to 2D zigzag coordinates `(x, y)` within strips of `num_steps` steps
/// of `step_length` (> 0) elements each.
pub fn swizzle(index: usize, num_steps: usize, #[comptime] step_length: u32) -> Coords2d {
    comptime!(assert!(
        step_length > 0,
        "swizzle: a strip holds at least one box"
    ));
    let full = num_steps as u32 * step_length;
    let index = index as u32;
    let strip_index = index / full;
    snake(
        strip_index,
        index - strip_index * full,
        num_steps,
        step_length,
        step_length * strip_index,
    )
}

/// The snake both swizzles walk.
#[cube]
fn snake(
    strip_index: u32,
    pos_in_strip: u32,
    num_steps: usize,
    width: u32,
    strip_offset: u32,
) -> Coords2d {
    let abs_step_index = pos_in_strip / width;
    let abs_pos_in_step = pos_in_strip % width;

    let strip_direction = strip_index % 2;
    let step_direction = abs_step_index % 2;

    let step_index = strip_direction * (num_steps as u32 - abs_step_index - 1)
        + (1 - strip_direction) * abs_step_index;
    let pos_in_step =
        step_direction * (width - abs_pos_in_step - 1) + (1 - step_direction) * abs_pos_in_step;

    (step_index, pos_in_step + strip_offset)
}

/// The swizzled cube level's two in-plane positions; zeros where no order is stated.
#[cube]
pub(crate) fn swizzled_positions(
    #[comptime] space: Space,
    #[comptime] level: Level,
    instances: &Coords<usize>,
) -> Coords<usize> {
    let mut out = Coords::<usize>::new();
    if comptime!(level.order().swizzles()) {
        let (x_at, y_at) = comptime!(in_plane_axes(&space, &level));
        let (count_x, count_y) = (instances.at(x_at), instances.at(y_at));
        let flat =
            CubeAxis::position(CubeAxis::X).plus(CubeAxis::position(CubeAxis::Y).times(count_x));
        let (x, y) = cube_positions(flat, (count_x, count_y), comptime!(level.order()));
        out.push(x);
        out.push(y);
    } else {
        out.push(0usize);
        out.push(0usize);
    }
    out
}

/// Where this space holds the cube level's two in-plane axes, `(x, y)`. Panics unless each
/// owns its whole grid dimension.
pub(crate) fn in_plane_axes(space: &Space, level: &Level) -> (usize, usize) {
    let at = |wanted: CubeAxis| {
        let found: Vec<usize> = (0..space.rank())
            .filter(|&p| level.cube_axis(space.axis_at(p)) == Some(wanted))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "Walk: a {:?} cube level distributes the grid's {wanted:?} dimension to {} of its axes, \
             and a swizzle can put back only an axis that owns a whole dimension",
            level.order(),
            found.len()
        );
        let p = found[0];
        assert_eq!(
            AxisDistribution::unspanned_weight(level, space, space.axis_at(p)),
            1,
            "Walk: a {:?} cube level shares the grid's {wanted:?} dimension with an axis this \
             space does not span, whose coordinate a swizzle has no way to put back",
            level.order()
        );
        p
    };
    (at(CubeAxis::X), at(CubeAxis::Y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every order is a permutation of the grid, divisible or ragged.
    #[test]
    fn every_order_is_a_permutation_of_the_grid() {
        for &(x, y) in &[
            (8usize, 8usize),
            (16, 4),
            (4, 16),
            (32, 8),
            (6, 10),
            (1, 7),
            (7, 5),
            (10, 3),
            (3, 9),
            (13, 1),
        ] {
            for order in [
                CubeOrder::RowMajor,
                CubeOrder::SwizzleRow(2),
                CubeOrder::SwizzleRow(4),
                CubeOrder::SwizzleCol(2),
                CubeOrder::SwizzleCol(8),
            ] {
                let mut seen = vec![false; x * y];
                for flat in 0..x * y {
                    let (px, py) = positions_host(flat, (x, y), order);
                    assert!(
                        px < x && py < y,
                        "{order:?} on {x}x{y}: ({px}, {py}) is off the grid"
                    );
                    let at = py * x + px;
                    assert!(
                        !seen[at],
                        "{order:?} on {x}x{y}: two cubes hold ({px}, {py})"
                    );
                    seen[at] = true;
                }
            }
        }
    }

    /// A strip one box wide is the grid's own order.
    #[test]
    fn a_strip_one_box_wide_is_the_grids_own_order() {
        assert_eq!(CubeOrder::SwizzleRow(1).canonicalize(), CubeOrder::RowMajor);
        assert_eq!(CubeOrder::SwizzleCol(1).canonicalize(), CubeOrder::RowMajor);
        assert!(!CubeOrder::SwizzleRow(1).swizzles());
        assert!(CubeOrder::SwizzleRow(4).swizzles());
    }

    /// An index past the grid lands off it without dividing by zero.
    #[test]
    fn an_index_past_the_grid_lands_off_it() {
        for index in [12, 18, 30] {
            let (_, along) = swizzle_ragged(index, 3, 2, 4);
            assert!(
                along >= 4,
                "index {index}, past the grid, landed on it at {along}"
            );
        }
    }

    /// Over a 16x16 grid, the first sixteen cubes of `SwizzleRow(4)` span four columns.
    #[test]
    fn a_swizzle_keeps_the_cubes_that_run_together_close() {
        let span = |order: CubeOrder| {
            let boxes: Vec<_> = (0..16)
                .map(|f| positions_host(f, (16, 16), order))
                .collect();
            let xs = boxes.iter().map(|b| b.0);
            xs.clone().max().unwrap() - xs.min().unwrap() + 1
        };
        assert_eq!(span(CubeOrder::RowMajor), 16);
        assert_eq!(span(CubeOrder::SwizzleRow(4)), 4);
    }

    /// The host twin of [`cube_positions`].
    fn positions_host(flat: usize, cubes: (usize, usize), order: CubeOrder) -> (usize, usize) {
        let (count_x, count_y) = cubes;
        match order.canonicalize() {
            CubeOrder::RowMajor => (flat % count_x, flat / count_x),
            CubeOrder::SwizzleRow(width) => {
                let (step, along) = swizzle_ragged(flat, count_y, width as u32, count_x);
                (along as usize, step as usize)
            }
            CubeOrder::SwizzleCol(width) => {
                let (step, along) = swizzle_ragged(flat, count_x, width as u32, count_y);
                (step as usize, along as usize)
            }
        }
    }
}
