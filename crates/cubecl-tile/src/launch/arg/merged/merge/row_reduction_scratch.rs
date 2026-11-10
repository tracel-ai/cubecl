//! The buffers a row reduction works through and leaves its values in: [`RowReductionScratch`].

use cubecl::client::Client;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, Type, UIntKind};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::unexpanded;

use super::box_holder::BoxHolder;
use super::row_reduction::RowReduction;
use crate::launch::Unbound;
use crate::launch::arg::merged::row_values::RowValues;
use crate::layout::RuntimeGeometry;
use crate::*;

/// What a [`ReducedRows`](crate::launch::ReducedRows) launch merges its rows through and the values it
/// leaves, one `f32` a row ([`values`](Self::values)). Made once for a launch shape and reused:
/// every launch overwrites the parts and leaves the counters at zero.
#[derive(Clone, Debug)]
pub struct RowReductionScratch {
    /// One part a row and box column.
    parts: Handle,
    /// One `u32` a row of boxes.
    counters: Handle,
    values: RowValues,
    rows: usize,
    columns: usize,
    /// The box columns: the holders a row's parts come from.
    holders: usize,
    vector_size: usize,
    /// The output's row and column axes.
    axes: (Axis, Axis),
    /// The kernel's partitioning, which says which boxes a unit's cube holds, and the launch's
    /// space that sizes its dynamic extents.
    partitioning: Partitioning,
    space: Space,
}

impl RowReductionScratch {
    /// The row sums of an output `out`, stored by `launcher`'s kernel.
    ///
    /// # Errors
    ///
    /// An output that is not a row-major matrix, a first level whose boxes are not each held whole
    /// by one cube or one plane, or, where a row is split between several boxes, a device on which
    /// cubes cannot hand each other their parts.
    pub fn new(launcher: &Launcher, out: &Unbound) -> Result<Self, Refusal> {
        let partitioning = launcher.partitioning();
        let axes = out.spec.axes();
        if axes.len() != 2 || !out.spec.projection.is_direct() || out.geometry.rank() != 2 {
            return Err(Refusal::Declined {
                why: "a row sum stores a matrix whose axes are its rows and columns",
            });
        }
        let Some(level) = partitioning.levels().first() else {
            return Err(Refusal::Declined {
                why: "a row sum's boxes are dealt by a first level, and this launch has none",
            });
        };
        BoxHolder::of(level)?;
        if level.axes().iter().any(|axis| !axes.contains(axis)) {
            return Err(Refusal::Declined {
                why: "a row sum's first level cuts the output's rows and columns alone",
            });
        }
        let (rows, columns, vector_size) = (
            out.geometry.shape()[0],
            out.geometry.shape()[1],
            out.vector_size,
        );
        let box_rows = level.tile(axes[0]).unwrap_or(rows);
        let box_columns = level.tile(axes[1]).unwrap_or(columns);
        if !box_columns.is_multiple_of(vector_size) || !columns.is_multiple_of(vector_size) {
            return Err(Refusal::Declined {
                why: "a row sum reads its boxes back in whole vectors",
            });
        }
        let holders = columns.div_ceil(box_columns);
        if holders > 1 {
            Self::require_hand_off(launcher.client())?;
        }
        let client = launcher.client();
        let row_boxes = rows.div_ceil(box_rows);
        let f32s = |count: usize| client.empty(count * size_of::<f32>());
        Ok(Self {
            parts: f32s(rows * holders),
            counters: client.create_from_slice(&vec![0u8; row_boxes * size_of::<u32>()]),
            values: RowValues::new(f32s(rows), rows),
            rows,
            columns,
            holders,
            vector_size,
            axes: (axes[0], axes[1]),
            partitioning: partitioning.clone(),
            space: launcher.space().clone(),
        })
    }

    /// Cubes hand each other their parts only where the runtime publishes one cube's writes to
    /// another within a dispatch, and counts atomically.
    fn require_hand_off(client: &Client) -> Result<(), Refusal> {
        let counts = client
            .properties()
            .atomic_type_usage(Type::atomic(ElemType::UInt(UIntKind::U32)));
        if !client.properties().features.device_memory_scope
            || !counts.contains(AtomicUsage::Add)
            || !counts.contains(AtomicUsage::LoadStore)
        {
            return Err(Refusal::Declined {
                why: "a row sum's cubes hand each other their parts, which this device does not \
                      promise within a dispatch",
            });
        }
        Ok(())
    }

    /// The value of each row, once a launch wrote them.
    pub fn values(&self) -> &RowValues {
        &self.values
    }

    /// The launch argument for the kernel that derived `out`; panics when the buffers were made
    /// for another output.
    pub(crate) fn arg_for(self, out: &Unbound) -> RowReductionScratchArgLaunch {
        let shape = out.geometry.shape();
        let axes = out.spec.axes();
        assert!(
            self.rows == shape[0]
                && self.columns == shape[1]
                && self.vector_size == out.vector_size
                && axes[..] == [self.axes.0, self.axes.1],
            "ReducedRows: the buffers were made for a {}x{} output over {:?} in vectors of {}, and the \
             kernel stores {shape:?} over {axes:?} in vectors of {}",
            self.rows,
            self.columns,
            self.axes,
            self.vector_size,
            out.vector_size,
        );
        let counters = self.counters.size() as usize / size_of::<u32>();
        // SAFETY: each handle was allocated for the count of `f32` or `u32` it is bound as.
        unsafe {
            RowReductionScratchArgLaunch::new(
                BufferArg::from_raw_parts(self.parts, self.rows * self.holders),
                BufferArg::from_raw_parts(self.counters, counters),
                self.values.arg(),
                self.partitioning.launch_arg(&self.space),
                self.vector_size,
                self.holders == 1,
                self.axes.0,
                self.axes.1,
            )
        }
    }
}

/// The device side of [`RowReductionScratch`].
#[derive(CubeType, CubeLaunch)]
pub struct RowReductionScratchArg {
    parts: &'static [f32],
    counters: &'static [Atomic<u32>],
    values: &'static [f32],
    /// The kernel's partitioning, which says which boxes a unit's cube holds.
    partitioning: Partitioning,
    /// The vector size the output is read back in.
    #[cube(comptime)]
    vector_size: usize,
    /// Each row lies in one box, so a holder's part is its row's whole sum.
    #[cube(comptime)]
    held_whole: bool,
    #[cube(comptime)]
    rows_axis: Axis,
    #[cube(comptime)]
    columns_axis: Axis,
}

#[cube]
impl RowReductionScratchArg {
    /// Reduce the rows of `out`, cut by the partitioning's first level, into the values. Every
    /// holder sums its box's part of each row. A row held whole by one box is done there; otherwise
    /// each holder parks its parts and the last of each row of boxes to count in adds them up.
    pub(crate) fn merge<E: Numeric, F: RowReduction>(
        &self,
        out: &Tensor<E>,
        geometry: &RuntimeGeometry,
        param: &F::Param,
    ) {
        let partitioning = &self.partitioning;
        let rows_axis = comptime!(self.rows_axis);
        let columns_axis = comptime!(self.columns_axis);
        let held_whole = comptime!(self.held_whole);
        let level = comptime!(partitioning.levels()[0].clone());
        let holder = comptime!(
            BoxHolder::of(&level).expect("RowReductionScratch::new admitted this launch")
        );
        let size!(W) = comptime!(self.vector_size);
        let (rows, columns) = (geometry.extent(0usize), geometry.extent(1usize));
        let row_stride = geometry.stride(0usize);
        let mut box_rows = rows;
        if let Some(tile) = comptime!(level.tile(rows_axis)) {
            box_rows = tile;
        }
        let mut box_columns = columns;
        if let Some(tile) = comptime!(level.tile(columns_axis)) {
            box_columns = tile;
        }
        let mut holders = 1usize;
        if comptime!(!held_whole) {
            holders = columns.div_ceil(box_columns);
        }

        // A plane sums one row at a time, its units taking turns along it. A cube's planes take
        // the rows of its box in turn, and a unit past the cube's whole planes takes none.
        let lanes = PLANE_DIM.min(CUBE_DIM) as usize;
        let lane = UNIT_POS_PLANE as usize;
        let mut group = 0usize;
        let mut groups = 1usize;
        let mut leads = lane == 0;
        if comptime!(holder == BoxHolder::Cube) {
            group = UNIT_POS as usize / lanes;
            groups = CUBE_DIM as usize / lanes;
            leads = UNIT_POS == 0;
        }

        // The cube's own stores are visible to every unit of it.
        sync_storage();
        for region in partitioning {
            let (first_row, first_column) = (region.origin(rows_axis), region.origin(columns_axis));
            if first_row < rows && first_column < columns && group < groups {
                let end_row = (first_row + box_rows).min(rows);
                let end_column = (first_column + box_columns).min(columns);
                let column = first_column / box_columns;
                let mut row = first_row + group;
                while row < end_row {
                    let start = row * row_stride;
                    let mut cell = start + first_column + lane * W::value();
                    let end = start + end_column;
                    let mut part = 0.0f32;
                    while cell < end {
                        let mut vector = Vector::<E, W>::empty();
                        #[unroll]
                        for k in 0..W::value() {
                            vector.insert(k, out[cell + k]);
                        }
                        part += Monoid::reduce::<f32, W>(
                            F::cells::<E, W>(vector),
                            W::value(),
                            Monoid::Sum,
                        );
                        cell += lanes * W::value();
                    }
                    let part = plane_sum(part);
                    if lane == 0 {
                        if comptime!(held_whole) {
                            write_f32(self.values, row, F::of_sum(part, param));
                        } else {
                            write_f32(self.parts, row * holders + column, part);
                        }
                    }
                    row += groups;
                }
            }
        }

        if comptime!(!held_whole) {
            // Unlike `LastArrival::hand_on`, which parks tile boxes a run of a shared level, this
            // parks one `f32` a row and box column.
            // The parts are published before any holder counts itself in.
            sync_storage();
            let mut arrived = Shared::<[u32]>::new_slice(1usize);
            let mut last = false;
            for region in partitioning {
                let (first_row, first_column) =
                    (region.origin(rows_axis), region.origin(columns_axis));
                if first_row < rows && first_column < columns {
                    let mut before = 0u32;
                    if leads {
                        before = self.counters[first_row / box_rows].fetch_add(1u32);
                    }
                    if comptime!(holder == BoxHolder::Cube) {
                        if leads {
                            arrived[0] = before;
                        }
                    } else {
                        last = plane_broadcast(before, 0u32) as usize + 1 == holders;
                    }
                }
            }
            // What the other holders parked is acquired before the last reads it.
            sync_storage();
            if comptime!(holder == BoxHolder::Cube) {
                last = arrived[0] as usize + 1 == holders;
            }

            for region in partitioning {
                let (first_row, first_column) =
                    (region.origin(rows_axis), region.origin(columns_axis));
                if first_row < rows && first_column < columns && last {
                    if group < groups {
                        let end_row = (first_row + box_rows).min(rows);
                        let mut row = first_row + group;
                        while row < end_row {
                            let mut part = 0.0f32;
                            let mut column = lane;
                            while column < holders {
                                part += self.parts[row * holders + column];
                                column += lanes;
                            }
                            let sum = plane_sum(part);
                            if lane == 0 {
                                write_f32(self.values, row, F::of_sum(sum, param));
                            }
                            row += groups;
                        }
                    }
                    if leads {
                        self.counters[first_row / box_rows].store(0u32);
                    }
                }
            }
        }
    }
}

/// Write `value` at `index` of `cells`. The fields are declared read-only because the `Merge`
/// methods that reach them take `&Arg`; this is the one place a kernel writes through them, to
/// `parts` and to `values`.
fn write_f32(_cells: &[f32], _index: usize, _value: f32) {
    unexpanded!()
}

mod write_f32 {
    use super::*;

    pub fn expand(
        scope: &Scope,
        cells: &<[f32] as CubeType>::ExpandType,
        index: NativeExpand<usize>,
        value: NativeExpand<f32>,
    ) {
        let mut cells = ExpandTypeClone::clone_unchecked(cells);
        store_cell::expand(scope, &mut cells, index, value);
    }
}

#[cube]
fn store_cell(cells: &mut [f32], index: usize, value: f32) {
    cells[index] = value;
}
