use core::marker::PhantomData;

use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::{CopyMechanism, Quantization};
use crate::matmul::components::global::CopyMechanism;
use crate::matmul::components::{InputIdent, MatmulPrecision};
use crate::matmul::components::{
    global::{self, GlobalConfig, single_stage, tensor_view::MappedTensorReader},
    stage::{self, ContiguousTilingLayout, RowMajorTilingOrder, Stage, multi_buffer::FullReader},
};

#[derive(CubeType)]
pub struct TmaLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub barrier: Barrier<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> TmaLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x: u32,
        y: u32,
        batch: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let stage = Stage::new_aligned::<G::SmmConfig>(
            comptime!(input_ident.as_ident()),
            128u32,
            config.to_smm_config(),
        );

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);
        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        TmaLoader::<MP, S> {
            tensor_view,
            barrier,
            stage,
            input_ident,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        _mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            let mut stage = this.stage.as_slice_mut().try_cast_unchecked();
            this.barrier.memcpy_async_tensor_to_shared_3d(
                &this.tensor_view.tensor,
                &mut stage,
                this.tensor_view.batch as i32,
                this.tensor_view.tile_y as i32,
                this.tensor_view.tile_x as i32,
            );
            this.barrier.arrive_tx(
                1,
                comptime!(
                    config.tiling_dimensions(this.input_ident).total_size() * MP::EI::elem_size()
                ),
            );
        } else {
            this.barrier.arrive();
        }
        this.barrier.wait();
    }

    pub fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(
            comptime!(this.input_ident.as_ident()),
            config.to_smm_config(),
        )
    }

    pub fn reader(this: &Self) -> FullReader<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>> {
        FullReader::new(this.stage, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view
            .update_view(k_offset, comptime!(this.input_ident.as_ident()));
    }
}
