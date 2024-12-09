use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;
use crate::matmul::components::global::tensor_view::TensorWriter;
use crate::matmul::components::global::tilewise_unloading::TilewiseUnloading;
use crate::matmul::components::stage::StageWriter;

use super::args::{GmmArgs, TensorOutput};

#[derive(CubeType)]
pub struct Unloader<GA: GmmArgs<EG>, EG: Numeric> {
    pub tensor_view: TensorWriter<GA, EG>,
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric> global::Unloader<EG> for Unloader<GA, EG> {
    type StageWriter = Self;

    fn as_stage_writer<G: global::Config>(this: Self) -> Self::StageWriter {
        this
    }
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric> Unloader<GA, EG> {
    pub fn new(
        tensor: TensorOutput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        Unloader::<GA, EG> {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric> StageWriter<EG> for Unloader<GA, EG> {
    fn write<ES: Numeric, G: global::Config>(
        this: &mut Self,
        slice: Slice<Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
        TilewiseUnloading::unload_from_slice::<GA, EG, ES, G>(
            &mut this.tensor_view,
            slice,
            compute_plane_offset,
            accumulator_offset,
            config,
        );
    }
}
