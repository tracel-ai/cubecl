use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{
    Layout, LayoutExpand,
    as_dyn::{IntoDyn, IntoDynExpand},
};

#[derive(CubeType, CubeLaunch)]
pub struct FixedDimLayout<D: IntoDyn> {
    shape: D,
    strides: Sequence<usize>,
    #[cube(comptime)]
    line_size: LineSize,
    #[cube(comptime)]
    checked: bool,
}

#[cube]
impl<D: IntoDyn> FixedDimLayout<D> {
    pub fn new(
        shape: D,
        strides: Sequence<usize>,
        #[comptime] line_size: LineSize,
        #[comptime] checked: bool,
    ) -> Self {
        FixedDimLayout::<D> {
            shape,
            strides,
            line_size,
            checked,
        }
    }
}

#[cube]
impl<D: IntoDyn> Layout for FixedDimLayout<D> {
    type Coordinates = D;
    type SourceCoordinates = usize;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let pos = pos.into_dyn();
        let mut offset = 0;

        #[unroll]
        for i in 0..pos.len() {
            offset += pos[i] as usize * self.strides[i];
        }

        offset / self.line_size
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut in_bounds = true;
        if comptime![self.checked] {
            let pos = pos.into_dyn();
            let shape = self.shape.clone().into_dyn();

            #[unroll]
            for i in 0..pos.len() {
                in_bounds &= pos[i] < shape[i];
            }
        }
        in_bounds
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos.clone()), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }
}

impl<'a, D: IntoDyn, R: Runtime> FixedDimLayoutLaunch<'a, D, R> {
    pub fn from_shape_handle(
        handle: &TensorBinding<R>,
        shape: D::RuntimeArg<'a, R>,
        line_size: LineSize,
    ) -> Self {
        let strides = handle.strides.iter().copied().map(ScalarArg::new).collect();
        Self::new(shape, strides, line_size, true)
    }

    pub fn from_shape_handle_unchecked(
        handle: &TensorBinding<R>,
        shape: D::RuntimeArg<'a, R>,
        line_size: LineSize,
    ) -> Self {
        let strides = handle.strides.iter().copied().map(ScalarArg::new).collect();
        Self::new(shape, strides, line_size, false)
    }
}
