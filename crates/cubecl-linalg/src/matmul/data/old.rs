use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

#[cube]
/// Behave like it's always row major layout
/// Hides stuff like transposed layout, or sliced tile layout, or line size
pub trait VirtualMemory<E: CubePrimitive>: CubeType {
    fn read_single(vm: &Self, row: u32, col: u32) -> E;
    fn write_single(vm: &mut Self, row: u32, col: u32, value: E);
    fn layout(vm: &Self) -> MatrixLayout;
}

#[derive(CubeType)]
pub struct TensorGmem<E: Numeric> {
    underlying: Tensor<Line<E>>,
    stride_row: u32,
    stride_col: u32,
    layout: MatrixLayout,
}
#[derive(CubeType)]
pub struct ArrayGmem<E: Numeric> {
    underlying: Array<Line<E>>,
    stride_row: u32,
    stride_col: u32,
    layout: MatrixLayout,
}
#[derive(CubeType)]
pub struct Smem {}

/// Carry layout because it's comptime and stride computations aren't
#[cube]
fn new_tensor_gmem<E: Numeric>(
    underlying: Tensor<Line<E>>,
    #[comptime] layout: MatrixLayout,
) -> TensorGmem<E> {
    let stride_row = underlying.stride(underlying.rank() - 2);
    let stride_col = underlying.stride(underlying.rank() - 1);
    TensorGmem::<E> {
        underlying,
        stride_row,
        stride_col,
        layout,
    }
}

#[cube]
impl<E: Numeric> VirtualMemory<Line<E>> for TensorGmem<E> {
    fn read_single(gmem: &Self, row: u32, col: u32) -> Line<E> {
        let position =
            (row * gmem.stride_row + col * gmem.stride_col) / gmem.underlying.line_size();

        gmem.underlying[position]
    }

    fn write_single(gmem: &mut Self, row: u32, col: u32, value: Line<E>) {
        let position =
            (row * gmem.stride_row + col * gmem.stride_col) / gmem.underlying.line_size();

        gmem.underlying[position] = value;
    }

    fn layout(vm: &Self) -> MatrixLayout {
        vm.layout
    }
}

#[cube]
fn new_array_gmem<E: Numeric>(
    underlying: Array<Line<E>>,
    shape: (u32, u32),
    #[comptime] layout: MatrixLayout,
) -> ArrayGmem<E> {
    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (shape.1, 1),
        MatrixLayout::ColMajor => (1, shape.0),
    };
    ArrayGmem::<E> {
        underlying,
        stride_row,
        stride_col,
        layout,
    }
}

#[cube]
impl<E: Numeric> VirtualMemory<Line<E>> for ArrayGmem<E> {
    fn read_single(gmem: &Self, row: u32, col: u32) -> Line<E> {
        let position =
            (row * gmem.stride_row + col * gmem.stride_col) / gmem.underlying.line_size();

        gmem.underlying[position]
    }

    fn write_single(gmem: &mut Self, row: u32, col: u32, value: Line<E>) {
        let position =
            (row * gmem.stride_row + col * gmem.stride_col) / gmem.underlying.line_size();

        gmem.underlying[position] = value;
    }

    fn layout(vm: &Self) -> MatrixLayout {
        vm.layout
    }
}
