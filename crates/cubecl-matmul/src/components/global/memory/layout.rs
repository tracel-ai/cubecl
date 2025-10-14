use cubecl::prelude::*;
use cubecl_common::quant::scheme::{QuantLevel, QuantScheme};
use cubecl_core::{self as cubecl};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Coords1d, Coords2d, Coords3d, Layout, LayoutExpand},
};

use crate::components::{MatmulProblem, MatrixLayout, global::memory::GlobalMemoryConfig};

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct SimpleTmaGlobalLayout {
    #[cube(comptime)]
    transposed: bool,
    shape: Coords3d,
}

#[cube]
impl SimpleTmaGlobalLayout {
    /// Creates a new 2D layout with the batch set to `nth_batch`.
    pub fn new(shape: Coords3d, #[comptime] layout: MatrixLayout) -> Self {
        let transposed = comptime![matches!(layout, MatrixLayout::ColMajor)];
        SimpleTmaGlobalLayout { shape, transposed }
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
        self.shape
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
    #[cube(comptime)]
    packing: u32,
}

#[cube]
impl BatchedGlobalLayout {
    /// Create a new batched global layout. `batch_shape` should be based on the output shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_strides: Sequence<u32>,
        batch_shape: Sequence<FastDivmod>,
        shape_row: u32,
        shape_col: u32,
        stride_row: u32,
        stride_col: u32,
        #[comptime] config: GlobalMemoryConfig,
        #[comptime] packing: u32,
    ) -> Self {
        BatchedGlobalLayout {
            batch_shape,
            batch_strides,
            rows: shape_row,
            cols: shape_col,
            stride_row,
            stride_col,
            config,
            packing,
        }
    }
}

#[cube]
impl Layout for BatchedGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.config.global_line_size];
        let (mut batch, row, col) = coords;

        let (row, col) = match comptime![self.config.matrix_layout] {
            MatrixLayout::RowMajor => (row, col / self.packing),
            MatrixLayout::ColMajor => (row / self.packing, col),
        };

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
            .zip(&handle.shape[..rank - 2])
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
            1,
        )
    }

    pub fn from_quantized_handle(
        client: &ComputeClient<R::Server, R::Channel>,
        values: &TensorHandleRef<'a, R>,
        scales: &TensorHandleRef<'a, R>,
        shape: &'a [usize],
        problem: &MatmulProblem,
        config: GlobalMemoryConfig,
        scheme: QuantScheme,
    ) -> (
        BatchedGlobalLayoutLaunch<'a, R>,
        BatchedGlobalScaleLayoutArgs<'a, R>,
    ) {
        let rank = values.shape.len();
        let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
        let values_layout = {
            let (stride_row, stride_col) = (values.strides[rank - 2], values.strides[rank - 1]);

            let batch_shape = problem
                .out_batches
                .iter()
                .map(|shape| FastDivmodArgs::new(client, *shape as u32))
                .collect();
            let batch_strides = values.strides[..rank - 2]
                .iter()
                .zip(&values.shape[..rank - 2])
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
                scheme.num_quants() as u32,
            )
        };

        let scales_layout = {
            let shape = (ScalarArg::new(rows as u32), ScalarArg::new(cols as u32));

            match scheme.level {
                QuantLevel::Tensor => BatchedGlobalScaleLayoutArgs::PerTensor { shape },
                QuantLevel::Block(block_size) => {
                    let [block_row, block_col] = block_size.as_dim();
                    let mut config = config;
                    config.global_line_size = 1;
                    let scales_layout =
                        BatchedGlobalLayoutLaunch::from_handle(client, scales, problem, config);
                    BatchedGlobalScaleLayoutArgs::BlockScaled(BlockScaledLayoutLaunch::new(
                        shape,
                        scales_layout,
                        (block_row as u32, block_col as u32),
                    ))
                }
            }
        };

        (values_layout, scales_layout)
    }
}

#[derive(CubeType, CubeLaunch)]
pub enum BatchedGlobalScaleLayout {
    PerTensor { shape: Coords2d },
    BlockScaled(BlockScaledLayout),
}

/// Workaround for enums not supporting `comptime`, should fix that in the future
#[derive(CubeType, CubeLaunch)]
pub struct BlockScaledLayout {
    shape: Coords2d,
    scales_layout: BatchedGlobalLayout,
    #[cube(comptime)]
    block_size: Coords2d,
}

#[cube]
impl Layout for BatchedGlobalScaleLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        match self {
            BatchedGlobalScaleLayout::PerTensor { .. } => 0u32.runtime(),
            BatchedGlobalScaleLayout::BlockScaled(layout) => {
                let BlockScaledLayout {
                    scales_layout,
                    block_size,
                    ..
                } = layout;

                let (batch, row, col) = coords;
                let (block_row, block_col) = block_size;
                let (row, col) = (row / block_row, col / block_col);
                scales_layout.to_source_pos((batch, row, col))
            }
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (u32, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        match self {
            BatchedGlobalScaleLayout::PerTensor { shape } => (u32::MAX.runtime(), shape.0, shape.1),
            BatchedGlobalScaleLayout::BlockScaled(layout) => {
                let (row, col) = layout.shape;
                (u32::MAX.runtime(), row, col)
            }
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        match self {
            BatchedGlobalScaleLayout::PerTensor { .. } => true.runtime(),
            BatchedGlobalScaleLayout::BlockScaled(layout) => {
                let (_, row, col) = pos;
                let l = &layout.scales_layout;
                let (rows, cols) = layout.shape;

                match comptime!((l.config.check_row_bounds, l.config.check_col_bounds)) {
                    (true, true) => row < rows && col < cols,
                    (true, false) => row < rows,
                    (false, true) => col < cols,
                    (false, false) => true,
                }
            }
        }
    }
}
