use cubecl::prelude::*;
use cubecl_common::quant::scheme::{QuantLevel, QuantScheme};
use cubecl_core::{self as cubecl};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{
        Coords1d, Coords2d, Coords3d, Layout, LayoutExpand, VirtualLayout, VirtualLayoutLaunch,
    },
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct GlobalLayoutConfig {
    pub matrix_layout: MatrixLayout,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
}

impl From<GlobalMemoryConfig> for GlobalLayoutConfig {
    fn from(gmem_config: GlobalMemoryConfig) -> Self {
        GlobalLayoutConfig {
            matrix_layout: gmem_config.matrix_layout,
            check_row_bounds: gmem_config.check_row_bounds,
            check_col_bounds: gmem_config.check_col_bounds,
        }
    }
}

/// Global layout that uses the last two dimensions and ignores all others.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct GlobalLayout {
    batch_layout: VirtualLayout<Coords1d, Coords1d>,
    rows: u32,
    cols: u32,

    stride_row: u32,
    stride_col: u32,

    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    packing: u32,
    #[cube(comptime)]
    config: GlobalLayoutConfig,
}

#[cube]
impl GlobalLayout {
    /// Create a new batched global layout. `batch_shape` should be based on the output shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_layout: VirtualLayout<Coords1d, Coords1d>,
        shape_row: u32,
        shape_col: u32,
        stride_row: u32,
        stride_col: u32,
        #[comptime] line_size: u32,
        #[comptime] packing: u32,
        #[comptime] config: GlobalLayoutConfig,
    ) -> Self {
        GlobalLayout {
            batch_layout,
            rows: shape_row,
            cols: shape_col,
            stride_row,
            stride_col,
            line_size,
            packing,
            config,
        }
    }
}

#[cube]
impl Layout for GlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        let line_size = comptime![self.line_size];
        let (batch, row, col) = coords;
        let batch_offs = self.batch_layout.to_source_pos(batch);

        let (row, col) = match comptime![self.config.matrix_layout] {
            MatrixLayout::RowMajor => (row, col / self.packing),
            MatrixLayout::ColMajor => (row / self.packing, col),
        };

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

impl<'a, R: Runtime> GlobalLayoutLaunch<'a, R> {
    pub fn from_handle(
        handle: &TensorHandleRef<'a, R>,
        line_size: u8,
        config: GlobalLayoutConfig,
    ) -> Self {
        let rank = handle.shape.len();
        let rows = handle.shape[rank - 2];
        let cols = handle.shape[rank - 1];
        let stride_row = handle.strides[rank - 2];
        let stride_col = handle.strides[rank - 1];

        GlobalLayoutLaunch::new(
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
            ScalarArg::new(stride_row as u32),
            ScalarArg::new(stride_col as u32),
            line_size as u32,
            1,
            config,
        )
    }

    pub fn from_handle_batched(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'a, R>,
        problem: &MatmulProblem,
        line_size: u8,
        config: GlobalLayoutConfig,
    ) -> Self {
        let rank = handle.shape.len();
        let rows = handle.shape[rank - 2];
        let cols = handle.shape[rank - 1];
        let stride_row = handle.strides[rank - 2];
        let stride_col = handle.strides[rank - 1];

        let batch_layout = BatchLayoutLaunch::from_handle(client, handle, problem);

        GlobalLayoutLaunch::new(
            VirtualLayoutLaunch::new::<BatchLayout>(batch_layout),
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
            ScalarArg::new(stride_row as u32),
            ScalarArg::new(stride_col as u32),
            line_size as u32,
            1,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_quantized_handle(
        client: &ComputeClient<R>,
        values: &TensorHandleRef<'a, R>,
        scales: &TensorHandleRef<'a, R>,
        shape: &'a [usize],
        problem: &MatmulProblem,
        scheme: QuantScheme,
        line_size: u8,
        config: GlobalLayoutConfig,
    ) -> (GlobalLayoutLaunch<'a, R>, GlobalScaleLayoutArgs<'a, R>) {
        let rank = values.shape.len();
        let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
        let values_layout = {
            let (stride_row, stride_col) = (values.strides[rank - 2], values.strides[rank - 1]);

            let batch_layout = BatchLayoutLaunch::from_handle(client, values, problem);

            GlobalLayoutLaunch::new(
                VirtualLayoutLaunch::new::<BatchLayout>(batch_layout),
                ScalarArg::new(rows as u32),
                ScalarArg::new(cols as u32),
                ScalarArg::new(stride_row as u32),
                ScalarArg::new(stride_col as u32),
                line_size as u32,
                scheme.num_quants() as u32,
                config,
            )
        };

        let scales_layout = {
            let shape = (ScalarArg::new(rows as u32), ScalarArg::new(cols as u32));

            match scheme.level {
                QuantLevel::Tensor => GlobalScaleLayoutArgs::PerTensor { shape },
                QuantLevel::Block(block_size) => {
                    let [block_row, block_col] = block_size.as_dim();
                    // Scales are never vectorized because we require that `block_size >= line_size * num_quants`.
                    let scales_layout =
                        GlobalLayoutLaunch::from_handle_batched(client, scales, problem, 1, config);
                    GlobalScaleLayoutArgs::BlockScaled(BlockScaledLayoutLaunch::new(
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
pub struct BatchLayout {
    batch_shape: Sequence<FastDivmod>,
    batch_strides: Sequence<u32>,
}

#[cube]
impl BatchLayout {
    pub fn new(batch_strides: Sequence<u32>, batch_shape: Sequence<FastDivmod>) -> Self {
        BatchLayout {
            batch_shape,
            batch_strides,
        }
    }
}

#[cube]
impl Layout for BatchLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut batch = pos;
        let mut batch_offs = 0;
        let batch_shape = self.batch_shape.rev();
        let batch_strides = self.batch_strides.rev();

        #[unroll]
        for i in 0..batch_shape.len() {
            let (rem, local_pos) = batch_shape.index(i).div_mod(batch);
            batch = rem;
            batch_offs += local_pos * *batch_strides.index(i);
        }

        batch_offs
    }

    fn shape(&self) -> Self::Coordinates {
        u32::MAX.runtime()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Layout that passed through the coordinates with no checks or modification.
#[derive(CubeType, CubeLaunch)]
pub struct NoopLayout {}

#[cube]
impl NoopLayout {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        NoopLayout {}
    }
}

#[cube]
impl Layout for NoopLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        pos
    }

    fn shape(&self) -> Self::Coordinates {
        u32::MAX.runtime()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

impl<'a, R: Runtime> BatchLayoutLaunch<'a, R> {
    pub fn from_handle(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'a, R>,
        problem: &MatmulProblem,
    ) -> Self {
        let rank = handle.shape.len();
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
        BatchLayoutLaunch::new(batch_shape, batch_strides)
    }
}

#[derive(CubeType, CubeLaunch)]
pub enum GlobalScaleLayout {
    PerTensor { shape: Coords2d },
    BlockScaled(BlockScaledLayout),
}

/// Workaround for enums not supporting `comptime`, should fix that in the future
#[derive(CubeType, CubeLaunch)]
pub struct BlockScaledLayout {
    shape: Coords2d,
    scales_layout: GlobalLayout,
    #[cube(comptime)]
    block_size: Coords2d,
}

#[cube]
impl BlockScaledLayout {
    pub fn new(
        shape: Coords2d,
        scales_layout: GlobalLayout,
        #[comptime] block_size: Coords2d,
    ) -> Self {
        BlockScaledLayout {
            shape,
            scales_layout,
            block_size,
        }
    }
}

#[cube]
impl Layout for GlobalScaleLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> u32 {
        match self {
            GlobalScaleLayout::PerTensor { .. } => 0u32.runtime(),
            GlobalScaleLayout::BlockScaled(layout) => {
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
            GlobalScaleLayout::PerTensor { shape } => (u32::MAX.runtime(), shape.0, shape.1),
            GlobalScaleLayout::BlockScaled(layout) => {
                let (row, col) = layout.shape;
                (u32::MAX.runtime(), row, col)
            }
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        match self {
            GlobalScaleLayout::PerTensor { .. } => true.runtime(),
            GlobalScaleLayout::BlockScaled(layout) => {
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
