use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        layout::{Coords1d, Coords2d, Coords3d, Layout, LayoutExpand},
        r#virtual::VirtualTensor,
    },
};

use crate::components::{MatmulProblem, MatrixLayout, global::memory::GlobalMemoryConfig};

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, Clone, Copy)]
pub struct SimpleGlobalLayout {
    rows: u32,
    stride_row: u32,
    columns: u32,
    stride_col: u32,
    batch_offset: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl SimpleGlobalLayout {
    /// Creates a new 2D layout starting at `batch_offset`.
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        batch_offset: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let rank = tensor.rank();

        SimpleGlobalLayout {
            rows: tensor.shape(rank - 2),
            stride_row: tensor.stride(rank - 2),
            columns: tensor.shape(rank - 1),
            stride_col: tensor.stride(rank - 1),
            batch_offset,
            config,
        }
    }
}

#[cube]
impl Layout for SimpleGlobalLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.config.global_line_size];
        let (row, col) = coords;
        let idx = self.batch_offset + row * self.stride_row + col * self.stride_col;

        idx / line_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.rows, self.columns)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (row, col) = pos;

        match comptime!((self.config.check_row_bounds, self.config.check_col_bounds)) {
            (true, true) => row < self.rows && col < self.columns,
            (true, false) => row < self.rows,
            (false, true) => col < self.columns,
            (false, false) => true,
        }
    }
}

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct SimpleTmaGlobalLayout {
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl SimpleTmaGlobalLayout {
    /// Creates a new 2D layout with the batch set to `nth_batch`.
    pub fn new(#[comptime] layout: MatrixLayout) -> Self {
        let transposed = comptime![matches!(layout, MatrixLayout::ColMajor)];
        SimpleTmaGlobalLayout { transposed }
    }
}

#[cube]
impl Layout for SimpleTmaGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords3d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords3d {
        let (batch, row, col) = coords;
        // Tensor maps are required to have a stride of 1 on the last dim, so their shape is
        // transposed for col-major matrices. Need to compensate by swapping the coordinates.
        if comptime![self.transposed] {
            (batch, col, row)
        } else {
            (batch, row, col)
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords3d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        // No need to bounds check TMA loads
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // No need to bounds check TMA loads
        true.runtime()
    }
}

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct BatchedGlobalLayout {
    batch_shape: Sequence<FastDivmod>,
    batch_strides: Sequence<u32>,

    rows: u32,
    cols: u32,

    stride_row: u32,
    stride_col: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl Layout for BatchedGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.config.global_line_size];
        let (mut batch, row, col) = coords;

        // This looks expensive to calculate each time, but the batch is constant across all loop
        // iterations, so it'll get pulled out by the compiler and only calculated once. It will
        // generate more code for unrolled loops, but should be fine.
        // TODO: VALIDATE WITH PROFILER
        let mut batch_offs = 0;
        let batch_shape = self.batch_shape.rev();
        let batch_strides = self.batch_strides.rev();

        #[unroll]
        for i in 0..batch_shape.len() {
            let (rem, local_pos) = batch_shape.index(i).div_mod(batch);
            batch = rem;
            batch_offs += local_pos * *batch_strides.index(i);
        }

        let idx = batch_offs + row * self.stride_row + col * self.stride_col;

        idx / line_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX.runtime(), self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, row, col) = pos;

        match comptime!((self.config.check_row_bounds, self.config.check_col_bounds)) {
            (true, true) => row < self.rows && col < self.cols,
            (true, false) => row < self.rows,
            (false, true) => col < self.cols,
            (false, false) => true,
        }
    }
}

impl<'a, R: Runtime> BatchedGlobalLayoutLaunch<'a, R> {
    pub fn from_handle(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'a, R>,
        problem: &MatmulProblem,
        config: GlobalMemoryConfig,
    ) -> Self {
        let rank = handle.shape.len();
        let rows = handle.shape[rank - 2];
        let cols = handle.shape[rank - 1];
        let stride_row = handle.strides[rank - 2];
        let stride_col = handle.strides[rank - 1];

        let batch_shape = problem
            .out_batches
            .iter()
            .map(|shape| FastDivmodArgs::new(client, *shape as u32))
            .collect();
        let batch_strides = handle.strides[..rank - 2]
            .iter()
            .zip(&problem.lhs_batches)
            .map(|(stride, shape)| if *shape == 1 { 0 } else { *stride })
            .map(|stride| ScalarArg::new(stride as u32))
            .collect();

        BatchedGlobalLayoutLaunch::new(
            batch_shape,
            batch_strides,
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
            ScalarArg::new(stride_row as u32),
            ScalarArg::new(stride_col as u32),
            config,
        )
    }
}
