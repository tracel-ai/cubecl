use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};

use crate::matmul::components::{Ident, MatrixLayout, TilingDimensions};

use super::StageConfig;

#[derive(CubeType)]
pub enum StageView<ES: Numeric> {
    Full(FullStageView<ES>),
    Buffer(BufferStageView<ES>),
}

#[derive(CubeType)]
pub struct FullStageView<ES: Numeric> {
    pub slice: SliceMut<Line<ES>>,
}

#[derive(CubeType)]
pub struct BufferStageView<ES: Numeric> {
    // ident: InputIdent -> says if we cut horizontally or vertically
    // buffer index: u32
    _es: PhantomData<ES>,
}

impl<ES: Numeric> StageView<ES> {
    pub fn num_slices<S: StageConfig>(&self, _ident: Ident, _config: S) -> u32 {
        unexpanded!()
    }

    pub fn slice_length<S: StageConfig>(&self, _ident: Ident, _config: S) -> u32 {
        unexpanded!()
    }

    pub fn stage_dim<S: StageConfig>(&self, _ident: Ident, _config: S) -> TilingDimensions {
        unexpanded!()
    }
}

impl<ES: Numeric> StageViewExpand<ES> {
    pub fn num_slices<S: StageConfig>(&self, ident: Ident, config: S) -> u32 {
        match self {
            StageViewExpand::Full(_) => {
                let matrix_layout = config.matrix_layout(ident);
                let tiling_dimensions = config.tiling_dimensions(ident);

                match matrix_layout {
                    MatrixLayout::RowMajor => tiling_dimensions.total_row(),
                    MatrixLayout::ColMajor => tiling_dimensions.total_col(),
                }
            }
            StageViewExpand::Buffer(_) => todo!(),
        }
    }

    pub fn slice_length<S: StageConfig>(&self, ident: Ident, config: S) -> u32 {
        match self {
            StageViewExpand::Full(_) => {
                let matrix_layout = config.matrix_layout(ident);
                let tiling_dimensions = config.tiling_dimensions(ident);
                let line_size = config.line_size(ident);

                comptime!(match matrix_layout {
                    MatrixLayout::RowMajor => tiling_dimensions.total_col() / line_size,
                    MatrixLayout::ColMajor => tiling_dimensions.total_row() / line_size,
                })
            }
            StageViewExpand::Buffer(_) => todo!(),
        }
    }

    pub fn stage_dim<S: StageConfig>(&self, ident: Ident, config: S) -> TilingDimensions {
        match self {
            StageViewExpand::Full(_) => config.tiling_dimensions(ident),
            StageViewExpand::Buffer(_) => todo!(),
        }
    }

    pub fn __expand_num_slices_method<S: StageConfig>(
        &self,
        _scope: &mut Scope,
        ident: Ident,
        config: S,
    ) -> u32 {
        Self::num_slices(&self, ident, config)
    }

    pub fn __expand_slice_length_method<S: StageConfig>(
        &self,
        _scope: &mut Scope,
        ident: Ident,
        config: S,
    ) -> u32 {
        Self::slice_length(&self, ident, config)
    }

    pub fn __expand_stage_dim_method<S: StageConfig>(
        &self,
        _scope: &mut Scope,
        ident: Ident,
        config: S,
    ) -> TilingDimensions {
        Self::stage_dim(&self, ident, config)
    }
}

#[cube]
impl<ES: Numeric> StageView<ES> {
    pub fn get_at(&self, index: u32) -> Line<ES> {
        match self {
            StageView::Full(full_stage_view) => full_stage_view.slice[index],
            StageView::Buffer(_) => {
                // TODO
                Line::cast_from(0)
            }
        }
    }

    pub fn set_at(&mut self, index: u32, value: Line<ES>) {
        match self {
            StageView::Full(full_stage_view) => full_stage_view.slice[index] = value,
            StageView::Buffer(_) => {
                // TODO
            }
        }
    }

    // TODO TEMP
    pub fn slice_mut(&mut self, start: u32, end: u32) -> SliceMut<Line<ES>> {
        match self {
            StageView::Full(full_stage_view) => full_stage_view.slice.slice_mut(start, end),
            StageView::Buffer(_) => {
                // TODO TEMP
                Array::new(0).to_slice_mut()
            }
        }
    }
}
