use core::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

use crate::matmul::components::{
    InputIdent,
    global::{self, GlobalConfig, Quantization, single_stage, tensor_view::MappedTensorReader},
    stage::{self, FullReader, Stage},
};
use crate::matmul::components::{MatmulPrecision, MatrixLayout};
use crate::matmul::components::{global::CopyMechanism, stage::StridedTilingLayout};

pub type StageReader<MP> = FullReader<<MP as MatmulPrecision>::ES, StridedTilingLayout>;

#[derive(CubeType)]
pub struct TmaLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: Stage<MP::ES, StridedTilingLayout>,
    #[cube(comptime)]
    ident: InputIdent,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct TmaRhsLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: Stage<MP::ES, StridedTilingLayout>,
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
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let stage =
            Stage::new_aligned::<G::SmmConfig>(ident.as_ident(), 128u32, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);

        TmaLoader::<MP, S> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        barrier: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            // The tensor map is encoded as the transposed shape, so we need to swap coordinates
            let (row, col) = match config.matrix_layout(comptime!(this.ident.as_ident())) {
                MatrixLayout::RowMajor => (this.tensor_view.tile_x, this.tensor_view.tile_y),
                MatrixLayout::ColMajor => (this.tensor_view.tile_y, this.tensor_view.tile_x),
            };

            let tensor = this.tensor_view.tensor.try_cast_unchecked();
            let mut stage = this.stage.as_slice_mut().try_cast_unchecked();

            CM::memcpy_async_tensor_to_shared_3d(
                barrier,
                &tensor,
                &mut stage,
                this.tensor_view.batch,
                row,
                col,
            );
        }
    }

    pub fn reader(this: &Self) -> StageReader<MP> {
        StageReader::<MP>::new(this.stage, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view
            .update_view(k_offset, comptime!(this.ident.as_ident()));
    }
}
