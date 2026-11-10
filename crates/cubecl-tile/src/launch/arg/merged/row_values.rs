//! The value a row reduction leaves for each row: [`RowValues`].

use cubecl::prelude::*;
use cubecl::server::Handle;

/// One `f32` a row of an output: what a [`ReducedRows`](crate::launch::ReducedRows) merge writes and
/// a [`RowScale`](crate::launch::RowScale) stage reads.
#[derive(Clone, Debug)]
pub struct RowValues {
    handle: Handle,
    rows: usize,
}

impl RowValues {
    /// `handle`, allocated for `rows` values of `f32`.
    pub(crate) fn new(handle: Handle, rows: usize) -> Self {
        Self { handle, rows }
    }

    /// The rows it holds a value for.
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn arg(self) -> BufferArg {
        // SAFETY: the handle was allocated for `rows` values of `f32`, the element every stage
        // and destination binding it reads.
        unsafe { BufferArg::from_raw_parts(self.handle, self.rows) }
    }
}
