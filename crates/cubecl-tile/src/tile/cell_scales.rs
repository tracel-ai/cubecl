//! The scales of a sum's cells where its holder keeps them ([`CellScales`]).

use std::collections::HashMap;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// A cell of a sum's window.
#[allow(dead_code)] // Read through its expand.
#[derive(CubeType, Clone)]
pub(crate) struct HeldCell {
    pub(crate) row: u32,
    pub(crate) col: u32,
    /// Where the cell lies in the window for the holder's first unit, where that is known as the
    /// kernel expands: two cells in one row or column for that unit are in one for every unit.
    #[cube(comptime)]
    pub(crate) first_unit: Option<(usize, usize)>,
}

/// The scales a sum's cells are under, read for the cells one unit holds. A scale is read once for
/// all the cells it Broadcasts to among those whose places are known as the kernel expands.
#[derive(Clone)]
pub(crate) struct CellScales;

/// The cells a scale value Broadcasts to: their row, where the scale varies along the rows, and
/// their column.
type Broadcasting = (Option<usize>, Option<usize>);

#[derive(Clone)]
pub(crate) struct CellScalesExpand {
    /// Each scale, read alone, by its index (Position).
    scales: Vec<TileExpand>,
    /// The sum's window and the dimensions its rows and columns lie along.
    sum: Space,
    dimensions: MatrixDimensions,
    /// What was read of each scale, by the scale and the cells it Broadcasts to (Keyed).
    read: HashMap<(usize, Broadcasting), NativeExpand<f32>>,
}

impl CellScales {
    /// How many scales a cell is under.
    fn len(&self) -> comptime_type!(usize) {
        unexpanded!()
    }

    /// Whether scale `scale` has one value for all of `cells`.
    fn broadcasts(&self, _scale: usize, _cells: &Sequence<HeldCell>) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// Scale `scale` of `cell`.
    fn at(&mut self, _scale: usize, _cell: &HeldCell) -> f32 {
        unexpanded!()
    }
}

#[cube]
impl CellScales {
    /// `line`, which holds `cells` a lane each, or one cell in every lane, under the scales.
    pub(crate) fn applied<T: Numeric, N: Size>(
        &mut self,
        line: Vector<T, N>,
        cells: &Sequence<HeldCell>,
    ) -> Vector<T, N> {
        let mut scaled = line;
        #[unroll]
        for s in 0..self.len() {
            let broadcasts = self.broadcasts(s, cells);
            let mut scale = Vector::<T, N>::empty();
            if broadcasts {
                scale = Vector::cast_from(self.at(s, cells.index(0usize)));
            } else {
                self.insert_each(&mut scale, s, cells);
            }
            scaled *= scale;
        }
        scaled
    }

    /// Scale `s` of each of `cells` into its lane of `scale`.
    fn insert_each<T: Numeric, N: Size>(
        &mut self,
        scale: &mut Vector<T, N>,
        #[comptime] s: usize,
        cells: &Sequence<HeldCell>,
    ) {
        #[unroll]
        for e in 0..cells.len() {
            scale.insert(e, T::cast_from(self.at(s, cells.index(e))));
        }
    }
}

impl CellScalesExpand {
    /// The scales of `factor`, for cells of a sum over `sum` whose rows and columns lie along
    /// `dimensions`. A scale with one value for every cell is read now.
    pub(crate) fn new(
        scope: &Scope,
        factor: &FactorExpand,
        sum: Space,
        dimensions: MatrixDimensions,
    ) -> Self {
        let mut scales = CellScalesExpand {
            scales: factor.each_alone(),
            sum,
            dimensions,
            read: HashMap::new(),
        };
        for scale in 0..scales.scales.len() {
            if scales.varies(scope, scale) == (false, false) {
                scales.read_at_origin(scope, scale);
            }
        }
        scales
    }

    /// Reads scale `scale` at the window's first cell, which holds it for every cell.
    fn read_at_origin(&mut self, scope: &Scope, scale: usize) {
        let first = HeldCellExpand {
            row: 0u32.into_expand(scope),
            col: 0u32.into_expand(scope),
            first_unit: Some((0, 0)),
        };
        self.__expand_at_method(scope, scale, &first);
    }

    /// Whether scale `scale` varies along the rows, and along the columns.
    fn varies(&self, scope: &Scope, scale: usize) -> (bool, bool) {
        let scale = &self.scales[scale];
        let addressed = |p: usize| scale.varies_along(scope, self.sum.dimension_at(p));
        let MatrixDimensions {
            row_split,
            col_split,
        } = self.dimensions;
        (
            (row_split..col_split).any(addressed),
            (col_split..self.sum.rank()).any(addressed),
        )
    }

    /// The cells reading what `cell` reads of scale `scale`, where its place is known.
    fn broadcasting(
        &self,
        scope: &Scope,
        scale: usize,
        cell: &HeldCellExpand,
    ) -> Option<Broadcasting> {
        let (along_rows, along_columns) = self.varies(scope, scale);
        let place = cell.first_unit;
        let row = along_rows.then_some(place.map(|(row, _)| row));
        let col = along_columns.then_some(place.map(|(_, col)| col));
        match (row, col) {
            (Some(None), _) | (_, Some(None)) => None,
            (row, col) => Some((row.flatten(), col.flatten())),
        }
    }

    fn __expand_len_method(&self, _scope: &Scope) -> usize {
        self.scales.len()
    }

    fn __expand_broadcasts_method(
        &self,
        scope: &Scope,
        scale: usize,
        cells: &SequenceExpand<HeldCell>,
    ) -> bool {
        let cells_read = (0..cells.__expand_len_method(scope)).map(|e| {
            let cell = cells.__expand_index_method(scope, e.into_expand(scope));
            self.broadcasting(scope, scale, cell)
        });
        let cells_read: Vec<_> = cells_read.collect();
        match cells_read.as_slice() {
            [_] => true,
            [first, others @ ..] => first.is_some() && others.iter().all(|other| other == first),
            [] => panic!("CellScales: a line holds a cell"),
        }
    }

    fn __expand_at_method(
        &mut self,
        scope: &Scope,
        scale: usize,
        cell: &HeldCellExpand,
    ) -> NativeExpand<f32> {
        let broadcasting = self.broadcasting(scope, scale, cell);
        if let Some(value) = broadcasting.and_then(|cells| self.read.get(&(scale, cells))) {
            return *value;
        }
        let value = scale_of_cell::expand(
            scope,
            &self.scales[scale],
            cell.row,
            cell.col,
            &self.sum,
            self.dimensions,
        );
        if let Some(cells) = broadcasting {
            self.read.insert((scale, cells), value);
        }
        value
    }
}

/// The value of `scale` at the cell `(row, col)` of a sum over `sum`.
#[cube]
fn scale_of_cell(
    scale: &Tile,
    row: u32,
    col: u32,
    #[comptime] sum: &Space,
    #[comptime] dimensions: MatrixDimensions,
) -> f32 {
    let at = TileMatrix::value_coords(row, col, 0usize, sum, dimensions, 1usize);
    scale.scale_for::<f32>(&at, sum, comptime!(Guard::Checked))
}

impl CubeType for CellScales {
    type ExpandType = CellScalesExpand;
}
impl IntoExpand for CellScalesExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for CellScalesExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl PlaceExpand for CellScalesExpand {
    fn create_mut_place(self, _: &Scope) -> Self {
        self
    }

    fn read_place(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for CellScalesExpand {}
impl AsRefExpand for CellScalesExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for CellScalesExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use cubecl::frontend::IntoExpand;
    use cubecl::ir::Scope;

    use super::*;
    use crate::test_scope::test_scope;

    const ROW: Dimension = Dimension(0);
    const COL: Dimension = Dimension(1);
    const BATCH: Dimension = Dimension(2);

    /// The scales of a batch of two `4 × 8` sums under one scale over `scale`, which varies along
    /// each of its dimensions.
    fn scales_over(scope: &Scope, scale: &[(Dimension, usize)]) -> CellScalesExpand {
        let sum = Space::new(&[(BATCH, 2), (ROW, 4), (COL, 8)]);
        let values = Tile::__expand_shared::<f32>(scope, sum.clone(), StageStorage::STRIDED);
        let scale = Tile::__expand_shared::<f32>(scope, Space::new(scale), StageStorage::STRIDED);
        let scaled = values.__expand_mul_method(scope, &scale);
        let dimensions = MatrixDimensions::trailing(&sum);
        CellScalesExpand::new(scope, &scaled.factor, sum, dimensions)
    }

    /// A cell whose place for the holder's first unit is `first_unit`.
    fn cell(scope: &Scope, first_unit: Option<(usize, usize)>) -> HeldCellExpand {
        HeldCellExpand {
            row: 0u32.into_expand(scope),
            col: 0u32.into_expand(scope),
            first_unit,
        }
    }

    fn cells(scope: &Scope, places: &[Option<(usize, usize)>]) -> SequenceExpand<HeldCell> {
        let mut cells = Sequence::<HeldCell>::__expand_new(scope);
        for &place in places {
            cells.__expand_push_method(scope, cell(scope, place));
        }
        cells
    }

    #[test]
    fn a_scale_varying_along_the_columns_broadcasts_to_cells_of_one_column() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(COL, 8)]);
        let column = cells(&scope, &[Some((0, 3)), Some((1, 3))]);

        assert!(scales.__expand_broadcasts_method(&scope, 0, &column));
    }

    #[test]
    fn a_scale_varying_along_the_columns_does_not_broadcast_to_cells_of_two_columns() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(COL, 8)]);
        let row = cells(&scope, &[Some((0, 3)), Some((0, 4))]);

        assert!(!scales.__expand_broadcasts_method(&scope, 0, &row));
    }

    #[test]
    fn a_varying_scale_does_not_broadcast_to_cells_of_unknown_places() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(COL, 8)]);
        let unknown = cells(&scope, &[None, None]);

        assert!(!scales.__expand_broadcasts_method(&scope, 0, &unknown));
    }

    #[test]
    fn a_scale_broadcasts_to_a_single_cell() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(ROW, 4), (COL, 8)]);
        let one = cells(&scope, &[None]);

        assert!(scales.__expand_broadcasts_method(&scope, 0, &one));
    }

    #[test]
    #[should_panic(expected = "a line holds a cell")]
    fn broadcasts_refuses_a_line_of_no_cell() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(COL, 8)]);

        scales.__expand_broadcasts_method(&scope, 0, &cells(&scope, &[]));
    }

    #[test]
    fn a_scale_varying_along_the_batch_alone_is_read_once_built() {
        let scope = test_scope();
        let scales = scales_over(&scope, &[(BATCH, 2)]);

        assert_eq!(scales.read.len(), 1);
    }

    #[test]
    fn a_scale_read_for_cells_it_broadcasts_to_is_read_once() {
        let scope = test_scope();
        let mut scales = scales_over(&scope, &[(COL, 8)]);

        scales.__expand_at_method(&scope, 0, &cell(&scope, Some((0, 3))));
        scales.__expand_at_method(&scope, 0, &cell(&scope, Some((1, 3))));

        assert_eq!(scales.read.len(), 1);
    }

    #[test]
    fn a_scale_read_for_a_cell_of_unknown_place_is_not_kept() {
        let scope = test_scope();
        let mut scales = scales_over(&scope, &[(COL, 8)]);

        scales.__expand_at_method(&scope, 0, &cell(&scope, None));

        assert!(scales.read.is_empty());
    }
}
