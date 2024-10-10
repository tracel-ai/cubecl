use super::Unloader;
use crate::matmul::matmul_global::new_array_view;
use crate::matmul::matmul_global::ArrayView;
use crate::matmul::matmul_stage::ArrayWriter;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct ArrayUnloader<E: Numeric> {
    pub view: ArrayView<E>,
    pub stage_info: StageInfo,
}

#[cube]
impl<E: Numeric> Unloader<E> for ArrayUnloader<E> {
    type GlobalView = ArrayView<E>;
    type StageWriter = ArrayWriter<E>;

    fn new(array: Array<Line<E>>, stage_info: StageInfo) -> Self {
        let shape_from_stage = (
            stage_info.num_tiles_x * stage_info.tile_size_x,
            stage_info.num_tiles_y * stage_info.tile_size_y,
        );
        let view = new_array_view(array, MatrixLayout::RowMajor, shape_from_stage);
        ArrayUnloader::<E> { view, stage_info }
    }

    fn unload(unloader: Self) -> Self::StageWriter {
        ArrayWriter::<E> {
            array_view: unloader.view,
            stage_info: unloader.stage_info,
        }
    }
}
