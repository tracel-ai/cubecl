use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::components::{
    global::memory::GlobalMemoryConfig,
    layout::{Coords2d, Layout},
};

/// Global layout that uses the last two dimensions and ignores all others
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
    type Coordinates = Coords2d;

    fn to_linear_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        coords.0 * this.stride_row + coords.1 * this.stride_col
    }

    fn to_linear_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let (row, col) = coords;
        let idx = row * this.stride_row + col * this.stride_col;

        let in_bounds =
            match comptime!((this.config.check_row_bounds, this.config.check_col_bounds)) {
                (true, true) => row < this.rows && col < this.columns,
                (true, false) => row < this.rows,
                (false, true) => col < this.columns,
                (false, false) => true,
            };

        (idx, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.rows, this.columns)
    }
}

mod r#virtual {
    use crate::components::layout::{VirtualLayout, VirtualLayoutOperationsExpand};

    use super::*;

    impl VirtualLayoutOperationsExpand<Coords2d> for SimpleGlobalLayoutExpand {
        fn __expand_to_linear_pos_method(
            &self,
            scope: &mut Scope,
            pos: <Coords2d as CubeType>::ExpandType,
        ) -> <u32 as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_to_linear_pos(scope, self.clone(), pos)
        }

        fn __expand_to_linear_pos_checked_method(
            &self,
            scope: &mut Scope,
            pos: <Coords2d as CubeType>::ExpandType,
        ) -> <(u32, bool) as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_to_linear_pos_checked(scope, self.clone(), pos)
        }

        fn __expand_shape_method(&self, scope: &mut Scope) -> <Coords2d as CubeType>::ExpandType {
            SimpleGlobalLayout::__expand_shape(scope, self.clone())
        }
    }

    #[cube]
    impl SimpleGlobalLayout {
        pub fn into_virtual(self) -> VirtualLayout<Coords2d> {
            VirtualLayout::new::<SimpleGlobalLayout>(self)
        }
    }
}
