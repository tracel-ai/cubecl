use super::base::Stage;
use super::TilingOrder;
use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_global::ReadView;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryStage<E: Numeric, O: TilingOrder> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    stage_info: StageInfo,
    line_size: u32,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<ES: Numeric, O: TilingOrder> Stage<ES> for SharedMemoryStage<ES, O> {
    type Underlying = SharedMemory<Line<ES>>;
    type Config = CmmaConfig;

    fn new(
        layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(stage_info) / line_size),
            line_size,
        );

        SharedMemoryStage::<ES, O> {
            smem,
            layout,
            stage_info: stage_info.runtime(),
            line_size,
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill<EG: Numeric, RV: ReadView<EG, Config = CmmaConfig>>(
        stage: &mut Self,
        gmem: &RV,
        config: Self::Config,
    ) {
        RV::load_shared_memory::<ES, O>(gmem, &mut stage.smem, stage.stage_info, config)
    }

    fn get_tile(stage: &Self, x: u32, y: u32) -> (&Slice<'_, Line<ES>>, MatrixLayout) {
        let nth_tile = O::to_nth_tile(
            x,
            y,
            stage.stage_info.num_tiles_x,
            stage.stage_info.num_tiles_y,
        );

        let tile_stride = tile_num_elements(stage.stage_info) / stage.line_size;
        let start = nth_tile * tile_stride;

        (stage.smem.slice(start, start + tile_stride), stage.layout)
    }

    fn layout(stage: &Self) -> MatrixLayout {
        stage.layout
    }
}
