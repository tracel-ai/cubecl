//! The order a plane level spanning the launch hands its tiles to the launch's planes
//! ([`PlaneOrder`]), and the cubes a unit level spanning the launch groups its units in
//! ([`UnitCube`]).

use super::{CubeOrder, Level};
use crate::{Dimension, Space};

/// The most cubes one grid axis takes on every device a launch may run on.
const CUBES_PER_GRID_DIMENSION: usize = 65_535;

/// The order the planes of a launch take a level's tiles in, where a plane level spans the launch
/// rather than one cube ([`Levels::launch_planes`](crate::Levels::launch_planes)): in blocks, each
/// a box of tiles held by consecutive planes, the blocks taken in a [`CubeOrder`] over their grid.
///
/// A plane of the launch is numbered by its cube and its place in it, `cube · planes_per_cube +
/// UNIT_POS_Y`, the cube read off the grid's `x` and `y`. Where a block holds a cube's planes,
/// as it does unless [`cubes_of`](PlaneOrder::cubes_of) says otherwise, each cube takes one block,
/// and the launch deals its tiles as a cube level with a plane level inside it would. Any order
/// is right: the planes share nothing, so the order only says which tiles the planes of one cube
/// take together.
///
/// Every plane the launch spawns walks one tile, as every plane of a cube level's cube does: a
/// plane past the last tile, of a block the edge cuts short or of a cube past the last block,
/// walks one past the grid, which the binding masks (`PlaneOrder::overruns`). So the
/// planes of a cube all reach any barrier one of them meets.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PlaneOrder {
    /// Tiles one block holds along each dimension it names; along a dimension it does not name,
    /// one.
    block: Vec<(Dimension, usize)>,
    block_order: CubeOrder,
    /// The launch's `y`.
    planes_per_cube: usize,
    /// At most this many cubes, their planes walking the tiles past the launch's grid-stride
    /// ([`capped`](PlaneOrder::capped)); every block a cube of its own where `None`.
    cubes: Option<usize>,
}

impl PlaneOrder {
    /// Blocks of `block` tiles, one tile a plane, taken in `block_order`, one block a cube: what a
    /// cube level holding a plane level of `block` hands out.
    pub fn new(block: &[(Dimension, usize)], block_order: CubeOrder) -> Self {
        for &(dimension, tiles) in block {
            assert!(
                tiles > 0,
                "PlaneOrder: a block holds no tile along {dimension:?}"
            );
        }
        let block = block.to_vec();
        let planes_per_cube = block.iter().map(|&(_, tiles)| tiles).product();
        Self {
            block,
            block_order: block_order.canonicalize(),
            planes_per_cube,
            cubes: None,
        }
    }

    /// This order launching at most `cubes` cubes: a plane then walks a tile, then the one a whole
    /// launch of planes further, until it runs past the last, as a persistent kernel's do. Fewer,
    /// longer-lived planes, where a launch of a plane a tile spends its time starting planes. The
    /// blocks are one cube's each and span one dimension.
    pub fn capped(self, cubes: usize) -> Self {
        assert!(
            cubes > 0,
            "PlaneOrder::capped: a launch holds at least one cube"
        );
        assert!(
            self.block.len() == 1,
            "PlaneOrder::capped: the planes walk the tiles of one dimension grid-stride, not {}",
            self.block.len()
        );
        Self {
            cubes: Some(cubes),
            ..self
        }
    }

    /// The most cubes the launch holds, where [`capped`](Self::capped).
    pub(crate) fn cap(&self) -> Option<usize> {
        self.cubes
    }

    /// This order with the launch's cubes holding `planes` planes each, which need not be a
    /// block's: a block then spans several cubes, or a cube several blocks.
    pub fn cubes_of(self, planes: usize) -> Self {
        assert!(
            planes > 0,
            "PlaneOrder::cubes_of: a cube holds at least one plane"
        );
        Self {
            planes_per_cube: planes,
            ..self
        }
    }

    pub fn planes_per_cube(&self) -> usize {
        self.planes_per_cube
    }

    /// Tiles one block holds along `dimension`.
    pub fn block_along(&self, dimension: Dimension) -> usize {
        self.block
            .iter()
            .find(|&&(a, _)| a == dimension)
            .map_or(1, |&(_, tiles)| tiles)
    }

    pub fn block_order(&self) -> CubeOrder {
        self.block_order
    }

    pub(crate) fn block_dimensions(&self) -> impl Iterator<Item = Dimension> + '_ {
        self.block.iter().map(|&(dimension, _)| dimension)
    }

    /// Planes one block holds, a tile each.
    pub(crate) fn block_planes(&self) -> usize {
        self.block.iter().map(|&(_, tiles)| tiles).product()
    }

    /// Whether each cube takes one block.
    pub(crate) fn one_block_a_cube(&self) -> bool {
        self.block_planes() == self.planes_per_cube
    }

    /// The grid's `x` and `y` for `level`'s blocks over `space`: every block's planes in cubes of
    /// [`planes_per_cube`](Self::planes_per_cube), laid along `x` and spilling into `y` past what
    /// one dimension takes. At least one cube, so an empty problem divides by no empty dimension.
    /// Static extents only.
    ///
    /// A spill lays the cubes out exactly where it can: `x` is the widest count one dimension
    /// takes that divides the cubes, so no cube runs past the last block. Only a count with no
    /// such divisor rounds `y` up, and leaves the spare cubes [`overruns`](Self::overruns) masks.
    pub(crate) fn grid(&self, level: &Level, space: &Space) -> (usize, usize) {
        let planes = self.blocks(level, space) * self.block_planes();
        let cubes = planes.div_ceil(self.planes_per_cube).max(1);
        let cubes = self.cubes.map_or(cubes, |cap| cubes.min(cap));
        if cubes <= CUBES_PER_GRID_DIMENSION {
            return (cubes, 1);
        }
        let rows = cubes.div_ceil(CUBES_PER_GRID_DIMENSION);
        let x = (rows..=CUBES_PER_GRID_DIMENSION)
            .find(|&rows| cubes.is_multiple_of(rows) && cubes / rows <= CUBES_PER_GRID_DIMENSION)
            .map_or(CUBES_PER_GRID_DIMENSION, |rows| cubes / rows);
        (x, cubes.div_ceil(x))
    }

    /// Whether some plane the launch spawns for `level` over `space` walks a tile past the last
    /// along `dimension`, so the binding masks it there: where the edge cuts a block short along
    /// `dimension`, or where the grid holds more planes than the blocks do and the blocks past the
    /// last land past `dimension`
    /// ([`block_coords`](crate::space::walk::launch_span::block_coords)). `dimension` is static;
    /// where an extent the grid counts is only the launch's, the planes may run past it, and it is
    /// masked.
    pub(crate) fn overruns(&self, level: &Level, space: &Space, dimension: Dimension) -> bool {
        let ragged = !level
            .tiles(space, dimension)
            .is_multiple_of(self.block_along(dimension));
        if ragged {
            return true;
        }
        // A capped launch's planes stop at the last tile rather than run past it.
        if self.cubes.is_some() {
            return false;
        }
        if !self.spare_blocks_land_past(level, space, dimension) {
            return false;
        }
        if level
            .dimensions()
            .into_iter()
            .any(|dimension| space.is_dynamic(dimension))
        {
            return true;
        }
        let (x, y) = self.grid(level, space);
        x * y * self.planes_per_cube > self.blocks(level, space) * self.block_planes()
    }

    /// Blocks `level` holds over `space`, the edge's counted.
    fn blocks(&self, level: &Level, space: &Space) -> usize {
        level
            .dimensions()
            .into_iter()
            .map(|dimension| {
                level
                    .tiles(space, dimension)
                    .div_ceil(self.block_along(dimension))
            })
            .product()
    }

    /// Whether a block index past the last can land past `dimension`
    /// ([`block_coords`](crate::space::walk::launch_span::block_coords)): the one dimension; of
    /// two, the one the order runs out along, and the other too where the order's strips leave a
    /// narrower last one, down which the index runs on; of more, the outermost, which is never
    /// wrapped.
    fn spare_blocks_land_past(&self, level: &Level, space: &Space, dimension: Dimension) -> bool {
        let dimensions = level.dimensions();
        match dimensions.len() {
            1 => dimension == dimensions[0],
            2 => match self.block_order.canonicalize() {
                CubeOrder::RowMajor => dimension == dimensions[1],
                CubeOrder::SwizzleRow(width) => {
                    dimension == dimensions[0]
                        || self.strips_are_ragged(level, space, dimensions[0], width)
                }
                CubeOrder::SwizzleCol(width) => {
                    dimension == dimensions[1]
                        || self.strips_are_ragged(level, space, dimensions[1], width)
                }
            },
            _ => dimension == dimensions[dimensions.len() - 1],
        }
    }

    /// Whether strips of `width` blocks along `across` leave a narrower last one, or may: the
    /// launch alone knows a dynamic extent's blocks.
    fn strips_are_ragged(
        &self,
        level: &Level,
        space: &Space,
        across: Dimension,
        width: usize,
    ) -> bool {
        space.is_dynamic(across)
            || !level
                .tiles(space, across)
                .div_ceil(self.block_along(across))
                .is_multiple_of(width)
    }
}

/// The cubes a unit level spanning the launch groups its units in
/// ([`Levels::launch_units`](crate::Levels::launch_units)): `planes` planes of `plane_width` units
/// each. The units share nothing, so the cube is only how the launch groups them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UnitCube {
    pub plane_width: usize,
    pub planes: usize,
}

impl UnitCube {
    /// Units one cube holds.
    pub fn units(&self) -> usize {
        self.plane_width * self.planes
    }

    /// The order the launch's units take a level's tiles in: one tile a unit, the tiles in row
    /// order, [`units`](Self::units) a cube.
    pub(crate) fn order(&self) -> PlaneOrder {
        PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(self.units())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::CubeCount;

    use crate::{Levels, Partitioning};

    const N: Dimension = Dimension(0);

    /// The grid's `x`, `y` and `z`.
    fn counts(partitioning: &Partitioning) -> (u32, u32, u32) {
        let CubeCount::Static(x, y, z) = partitioning.cube_count() else {
            unreachable!("a static space lays a static grid");
        };
        (x, y, z)
    }

    /// One plane a tile of two along `N`, one plane a cube, over `columns`.
    fn one_plane_a_cube(columns: usize) -> Partitioning {
        Partitioning::new(
            Space::new(&[(N, columns)]),
            Levels::leaf(&[(N, 2)])
                .launch_planes(&[N], PlaneOrder::new(&[(N, 1)], CubeOrder::RowMajor))
                .build(),
        )
    }

    /// Over blocks whose last strip is narrower, spare planes run past the strips' far end too.
    #[test]
    fn spare_planes_overrun_the_far_end_of_a_ragged_strip() {
        const M: Dimension = Dimension(1);
        let partitioning = Partitioning::new(
            Space::new(&[(M, 1), (N, 3)]),
            Levels::leaf(&[(M, 1), (N, 1)])
                .launch_planes(
                    &[M, N],
                    PlaneOrder::new(&[], CubeOrder::SwizzleRow(2)).cubes_of(2),
                )
                .build(),
        );
        assert!(partitioning.overhangs(N));
    }

    /// A capped launch's planes walk one dimension: two are refused on the host.
    #[test]
    #[should_panic(expected = "a capped launch's cubes take a block each")]
    fn a_capped_launch_over_two_dimensions_is_refused() {
        const M: Dimension = Dimension(1);
        Levels::leaf(&[(M, 1), (N, 1)])
            .launch_planes(
                &[M, N],
                PlaneOrder::new(&[(M, 2)], CubeOrder::RowMajor).capped(4),
            )
            .build();
    }

    /// More cubes than one dimension takes are laid out exactly where a count one dimension takes
    /// divides them: a vocabulary of columns, two a plane, spills into a second row of the same
    /// width, and no plane runs past the last tile. A count with no such divisor, a prime past
    /// one dimension, still spills into a row it does not fill, and overruns it.
    #[test]
    fn a_spill_divides_the_cubes_where_it_can() {
        let vocabulary = one_plane_a_cube(151_936);
        assert_eq!(counts(&vocabulary), (37_984, 2, 1));
        assert!(!vocabulary.overhangs(N));

        let prime = one_plane_a_cube(2 * 65_537);
        assert_eq!(counts(&prime), (CUBES_PER_GRID_DIMENSION as u32, 2, 1));
        assert!(prime.overhangs(N));
    }

    /// A grid one dimension holds stays one row.
    #[test]
    fn a_grid_one_dimension_holds_is_one_row() {
        assert_eq!(counts(&one_plane_a_cube(2 * 1000)), (1000, 1, 1));
    }
}
