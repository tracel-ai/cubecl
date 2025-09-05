use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{
    layout::{Coords1d, Coords3d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand},
    r#virtual::VirtualTensor,
};

use crate::components::global::memory::GlobalMemoryConfig;

/// Global layout that uses the last two dimensions and ignores all others.
/// Batch dim is treated as unit stride, and batch shape is always `1`
#[derive(CubeType, Clone, Copy)]
pub struct SimpleGlobalLayout {
    rows: u32,
    stride_row: u32,
    columns: u32,
    stride_col: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl SimpleGlobalLayout {
    pub fn new<T: Numeric, IO: Clone>(
        tensor: &VirtualTensor<T, IO>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self {
        let rank = tensor.rank();

        SimpleGlobalLayout {
            rows: tensor.shape(rank - 2),
            stride_row: tensor.stride(rank - 2),
            columns: tensor.shape(rank - 1),
            stride_col: tensor.stride(rank - 1),
            config,
        }
    }
}

#[cube]
impl Layout for SimpleGlobalLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        let (b, row, col) = coords;
        let idx = b + row * this.stride_row + col * this.stride_col;
        idx / comptime![this.config.global_line_size]
    }

    fn to_source_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        (this.to_source_pos(coords), this.is_in_bounds(coords))
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (1, this.rows, this.columns)
    }

    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool {
        let (_, row, col) = pos;

        match comptime!((this.config.check_row_bounds, this.config.check_col_bounds)) {
            (true, true) => row < this.rows && col < this.columns,
            (true, false) => row < this.rows,
            (false, true) => col < this.columns,
            (false, false) => true,
        }
    }
}

mod r#virtual {
    use cubecl_std::tensor::layout::*;

    use super::*;

    impl VirtualLayoutOperationsExpand<Coords3d, Coords1d> for SimpleGlobalLayoutExpand {
        fn __expand_to_source_pos_method(
            &self,
            scope: &mut Scope,
            pos: <Coords3d as CubeType>::ExpandType,
        ) -> <u32 as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_to_source_pos(scope, self.clone(), pos)
        }

        fn __expand_to_source_pos_checked_method(
            &self,
            scope: &mut Scope,
            pos: <Coords3d as CubeType>::ExpandType,
        ) -> <(u32, bool) as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_to_source_pos_checked(scope, self.clone(), pos)
        }

        fn __expand_shape_method(&self, scope: &mut Scope) -> <Coords3d as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_shape(scope, self.clone())
        }

        fn __expand_is_in_bounds_method(
            &self,
            scope: &mut Scope,
            pos: <Coords3d as CubeType>::ExpandType,
        ) -> ExpandElementTyped<bool> {
            SimpleGlobalLayout::__expand_is_in_bounds(scope, self.clone(), pos)
        }
    }

    #[cube]
    impl SimpleGlobalLayout {
        pub fn virt(self) -> VirtualLayout<Coords3d, Coords1d> {
            VirtualLayout::new::<SimpleGlobalLayout>(self)
        }
    }
}
