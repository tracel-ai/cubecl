//! The scales of a sum's cells where its holder keeps them ([`CellScales`]).

use std::collections::HashMap;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
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
    /// Each scale, read alone.
    scales: Vec<TileExpand>,
    /// The sum's window and the dimensions its rows and columns lie along.
    sum: Space,
    dimensions: MatrixDimensions,
    /// What was read of each scale.
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
                #[unroll]
                for e in 0..cells.len() {
                    scale.insert(e, T::cast_from(self.at(s, cells.index(e))));
                }
            }
            scaled *= scale;
        }
        scaled
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
                let first = HeldCellExpand {
                    row: 0u32.into_expand(scope),
                    col: 0u32.into_expand(scope),
                    first_unit: Some((0, 0)),
                };
                scales.__expand_at_method(scope, scale, &first);
            }
        }
        scales
    }

    /// Whether scale `scale` varies along the rows, and along the columns.
    fn varies(&self, scope: &Scope, scale: usize) -> (bool, bool) {
        let scale = &self.scales[scale];
        let projection = scale.clone().__expand_projection_method(scope);
        let addressed = |p: usize| {
            let dimension = self.sum.dimension_at(p);
            scale.place.space.contains(dimension) && projection.addresses(dimension)
        };
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
impl IntoMut for CellScalesExpand {
    fn into_mut(self, _: &Scope) -> Self {
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
