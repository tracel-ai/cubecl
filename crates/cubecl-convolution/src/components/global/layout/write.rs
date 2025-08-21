use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    global::memory::GlobalMemoryConfig,
    layout::{Coords2d, Layout},
};
use cubecl_std::{
    FastDivmod,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use crate::components::global::{layout::unwrap, load::im2col_tma::div_mod_seq};

#[derive(CubeType, Clone)]
pub struct NhwcOutGlobalLayout {
    pub stride_n: u32,
    pub strides_spatial: Sequence<u32>,
    pub stride_c: u32,

    pub shape_out: Sequence<FastDivmod>,

    pub shape_m: u32,
    pub shape_n: u32,

    #[cube(comptime)]
    pub config: GlobalMemoryConfig,
}

#[cube]
impl NhwcOutGlobalLayout {
    pub fn new<E: Numeric>(
        tensor: &VirtualTensor<E, ReadWrite>,
        shape_m: u32,
        shape_n: u32,
        shape_out: Sequence<FastDivmod>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> NhwcOutGlobalLayout {
        let spatial_dims = comptime![shape_out.len()];
        let mut strides_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
        }

        let stride_n = tensor.stride(0);
        let stride_c = tensor.stride(spatial_dims + 1);

        NhwcOutGlobalLayout {
            stride_n,
            strides_spatial,
            stride_c,
            shape_out,
            shape_m,
            shape_n,
            config,
        }
    }
}

#[cube]
impl Layout for NhwcOutGlobalLayout {
    type Coordinates = Coords2d;

    fn to_linear_pos(this: &Self, coords: Self::Coordinates) -> u32 {
        let (view_m, view_n) = coords;

        let (n, out_pos) = div_mod_seq(view_m, &this.shape_out);

        let spatial_dims = comptime![this.shape_out.len()];
        let c = view_n;

        let mut write_pos = n * this.stride_n + c * this.stride_c;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            write_pos += *out_pos.index(i) as u32 * *this.strides_spatial.index(i);
        }

        write_pos
    }

    fn to_linear_pos_checked(this: &Self, coords: Self::Coordinates) -> (u32, bool) {
        let linear_pos = Self::to_linear_pos(this, coords);

        let (m, n) = coords;
        let check_m = comptime![this.config.check_row_bounds];
        let check_n = comptime![this.config.check_col_bounds];
        let in_bounds = (!check_m || m < this.shape_m) && (!check_n || n < this.shape_n);

        (linear_pos, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        (this.shape_m, this.shape_n)
    }
}

mod r#virtual {
    use cubecl_matmul::components::layout::{VirtualLayout, VirtualLayoutOperationsExpand};

    use super::*;

    impl VirtualLayoutOperationsExpand<Coords2d> for NhwcOutGlobalLayoutExpand {
        fn __expand_to_linear_pos_method(
            &self,
            scope: &mut Scope,
            pos: <Coords2d as CubeType>::ExpandType,
        ) -> <u32 as CubeType>::ExpandType {
            NhwcOutGlobalLayout::__expand_to_linear_pos(scope, self.clone(), pos)
        }

        fn __expand_to_linear_pos_checked_method(
            &self,
            scope: &mut Scope,
            pos: <Coords2d as CubeType>::ExpandType,
        ) -> <(u32, bool) as CubeType>::ExpandType {
            NhwcOutGlobalLayout::__expand_to_linear_pos_checked(scope, self.clone(), pos)
        }

        fn __expand_shape_method(&self, scope: &mut Scope) -> <Coords2d as CubeType>::ExpandType {
            NhwcOutGlobalLayout::__expand_shape(scope, self.clone())
        }
    }

    #[cube]
    impl NhwcOutGlobalLayout {
        pub fn into_virtual(self) -> VirtualLayout<Coords2d> {
            VirtualLayout::new::<NhwcOutGlobalLayout>(self)
        }
    }
}
