use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    MatrixPrecision, MatmulIdent, StageIdent, stage::StageMemoryConfig,
};
use cubecl_std::FastDivmod;

use cubecl_matmul::components::stage::RowMajorTilingOrder;
use cubecl_matmul::components::{
    global::memory::MappedTensorReader,
    stage::{ContiguousTilingLayout, StageConfig, StridedStage},
};

use crate::kernels::layered::selector::RuntimeArgs;

pub type TmaWeightTiling = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type TmaWeightStage<IP> = StridedStage<<IP as MatrixPrecision>::Stage, TmaWeightTiling>;

#[derive(CubeType)]
pub struct TmaWeightGlobalReader<IP: MatrixPrecision, S: StageConfig> {
    pub tensor_view: MappedTensorReader<IP::Global>,
    pub stages: Sequence<StridedStage<IP::Stage, TmaWeightTiling>>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<IP: MatrixPrecision, S: StageConfig> TmaWeightGlobalReader<IP, S> {
    pub fn new(
        tensor: TensorMap<IP::Global>,
        x: u32,
        y: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] num_stages: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        let mut stages = Sequence::new();

        #[unroll]
        for _ in 0..num_stages {
            stages.push(StridedStage::new_aligned(StageIdent::Rhs, 128u32, config));
        }

        let tensor_view = MappedTensorReader::new(tensor, x, y, 0);

        TmaWeightGlobalReader::<IP, S> {
            tensor_view,
            stages,
            padded_channels: runtime_args.padded_channels,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage(
        &mut self,
        barrier: &Barrier,
        #[comptime] stage_idx: u32,
        #[comptime] config: S,
    ) {
        let stage = self.stages.index_mut(stage_idx);

        if UNIT_POS == 0 {
            let k = self.tensor_view.tile_x;
            let out_c = self.tensor_view.tile_y;

            let tensor = self.tensor_view.tensor.try_cast_unchecked();
            let mut stage = stage.as_slice_mut(1u32);
            let slice_size = config.tiling_scheme().elements_in_stage_n()
                * config.tiling_scheme().elements_in_tile_k();

            #[unroll]
            for tile_k in 0..config.tiling_scheme().tiles_in_stage_k() {
                let slice_start = slice_size * tile_k;
                let mut slice = stage.slice_mut(slice_start, slice_size);

                let k = k + tile_k * config.tiling_scheme().elements_in_tile_k();
                let (k_idx, in_c) = self.padded_channels.div_mod(k);

                barrier.tma_load_3d(&tensor, &mut slice, out_c as i32, k_idx as i32, in_c as i32);
            }
        }
    }

    pub fn stage(&self, #[comptime] stage_idx: u32) -> TmaWeightStage<IP> {
        *self.stages.index(stage_idx)
    }

    pub fn advance_view(&mut self, k_offset: u32) {
        self.tensor_view.update_view(k_offset, MatmulIdent::Rhs);
    }
}
