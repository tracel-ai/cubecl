use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{MatmulIdent, StageIdent};
use cubecl_std::{CubeOption, FastDivmod};

use crate::base::RuntimeArgs;
use cubecl_matmul::components::stage::RowMajorTilingOrder;
use cubecl_matmul::components::{
    MatmulPrecision, global::Quantization, stage::FullStageToTileReader,
};
use cubecl_matmul::components::{
    global::{self, global_memory::MappedTensorReader},
    stage::{ContiguousTilingLayout, StageConfig, StageMemory},
};

pub type TmaWeightTiling = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type TmaWeightReader<MP> = FullStageToTileReader<<MP as MatmulPrecision>::ES, TmaWeightTiling>;

#[derive(CubeType)]
pub struct TmaWeightLoader<MP: MatmulPrecision, S: StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stages: Sequence<StageMemory<MP::ES, TmaWeightTiling>>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: StageConfig> TmaWeightLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x: u32,
        y: u32,
        quantization: CubeOption<Quantization<MP>>,
        runtime_args: &RuntimeArgs,
        #[comptime] num_stages: u32,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let mut stages = Sequence::new();

        #[unroll]
        for _ in 0..num_stages {
            stages.push(StageMemory::new_aligned::<G::StageConfig>(
                StageIdent::Rhs,
                128u32,
                config.stage_config(),
            ));
        }

        let tensor_view = MappedTensorReader::new(tensor, x, y, 0);

        TmaWeightLoader::<MP, S> {
            tensor_view,
            stages,
            padded_channels: runtime_args.padded_channels,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage(
        this: &mut Self,
        barrier: &Barrier<MP::ES>,
        #[comptime] stage_idx: u32,
        #[comptime] config: S,
    ) {
        let stage = this.stages.index_mut(stage_idx);

        if UNIT_POS == 0 {
            let k = this.tensor_view.tile_x;
            let out_c = this.tensor_view.tile_y;

            let tensor = this.tensor_view.tensor.try_cast_unchecked();
            let mut stage = stage.as_slice_mut(1u32);
            let slice_size = config.tiling_scheme().elements_in_stage_n()
                * config.tiling_scheme().elements_in_tile_k();

            #[unroll]
            for tile_k in 0..config.tiling_scheme().tiles_in_stage_k() {
                let slice_start = slice_size * tile_k;
                let mut slice = stage.slice_mut(slice_start, slice_size);

                let k = k + tile_k * config.tiling_scheme().elements_in_tile_k();
                let (k_idx, in_c) = this.padded_channels.div_mod(k);

                barrier.tma_load_3d(&tensor, &mut slice, out_c as i32, k_idx as i32, in_c as i32);
            }
        }
    }

    pub fn reader(this: &Self, #[comptime] stage_idx: u32) -> TmaWeightReader<MP> {
        TmaWeightReader::<MP>::new(*this.stages.index(stage_idx), StageIdent::Rhs)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, MatmulIdent::Rhs);
    }
}
