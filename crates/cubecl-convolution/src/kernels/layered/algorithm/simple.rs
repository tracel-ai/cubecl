use cubecl_core::server::LaunchError;
use cubecl_core::{Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef};
use cubecl_matmul::components::{
    MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError, stage::StridedStageFamily,
    tile::io::Strided,
};
use cubecl_matmul::components::{
    global::args::TensorArgs, stage::PlaneMatmulFamily, tile::TileMatmulFamily,
};
use cubecl_matmul::components::{
    global::read::sync_full_cyclic::SyncFullCyclicLoading,
    stage::{ColMajorTilingOrder, NumStages, RowMajorTilingOrder},
};
use cubecl_std::{
    CubeOption,
    tensor::{TensorHandle, into_contiguous_pitched},
};
use std::marker::PhantomData;

use crate::components::{
    ConvolutionProblem, convolution_matmul_selection,
    global::{
        read::full_reader::FullLoadingStrategy, single_stage::simple::SimpleConvolutionFamily,
    },
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConvAlgorithm<
    TMM: TileMatmulFamily,
    LL: FullLoadingStrategy = SyncFullCyclicLoading<RowMajorTilingOrder>,
    LR: FullLoadingStrategy = SyncFullCyclicLoading<ColMajorTilingOrder>,
> {
    _tmm: PhantomData<TMM>,
    _loader: PhantomData<(LL, LR)>,
}

impl<
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
    LL: FullLoadingStrategy,
    LR: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
> Algorithm for SimpleConvAlgorithm<TMM, LL, LR>
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        Option<StridedStageFamily>,
    >;
    type GlobalConvolution = SimpleConvolutionFamily<Self::StageMatmul, LL, LR>;

    type Args = TensorArgs;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
    ) -> Result<TensorHandle<R>, LaunchError> {
        if has_valid_layout(handle) {
            Ok(TensorHandle::from_ref(handle, dtype))
        } else {
            into_contiguous_pitched(client, handle, dtype)
        }
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(convolution_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            TMM::should_swizzle(client),
            line_sizes,
            dtypes,
        )?)
    }
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    handle.strides[dim_c] == 1
}
