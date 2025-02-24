use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::tensor_view::TensorWriter;
use crate::matmul::components::global::tilewise_unloading::TilewiseUnloading;
use crate::matmul::components::stage::StageWriter;
use crate::matmul::components::Ident;
use crate::tensor::ReadWrite;
use crate::{matmul::components::global, tensor::VirtualTensor};

use super::args::Quantization;

#[derive(CubeType)]
pub struct Unloader<EG: Numeric> {
    pub tensor_view: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> global::OutputLoader<EG> for Unloader<EG> {
    type StageWriter = Self;

    fn as_stage_writer<G: global::GlobalConfig>(this: Self) -> Self::StageWriter {
        this
    }
}

#[cube]
impl<EG: Numeric> Unloader<EG> {
    pub fn new(
        tensor: VirtualTensor<EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        Unloader::<EG> {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for Unloader<EG> {
    fn write<ES: Numeric, G: global::GlobalConfig>(
        this: &mut Self,
        slice: SliceMut<Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
        TilewiseUnloading::unload_from_slice::<EG, ES, G>(
            &mut this.tensor_view,
            slice.to_slice(),
            compute_plane_offset,
            accumulator_offset,
            config,
        );
    }
}

// TODO: Merge with the main Unloaded.
#[derive(CubeType)]
pub struct UnloaderQuantized {
    pub tensor_view: TensorWriter<u8>,
    pub lhs_sums: SharedMemory<i32>, // TODO Use Line
    pub rhs_sums: SharedMemory<i32>,
    pub quantization: Quantization,
}

#[cube]
impl global::OutputLoader<u8> for UnloaderQuantized {
    type StageWriter = Self;

    fn as_stage_writer<G: global::GlobalConfig>(this: Self) -> Self::StageWriter {
        this
    }
}

#[cube]
impl UnloaderQuantized {
    pub fn new(
        tensor: VirtualTensor<u8, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: Quantization,
        #[comptime] out_shape: (u32, u32),
    ) -> Self {
        UnloaderQuantized {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
            lhs_sums: SharedMemory::new(out_shape.0),
            rhs_sums: SharedMemory::new(out_shape.1),
            quantization,
        }
    }
}

#[cube]
impl StageWriter<u8> for UnloaderQuantized {
    // ES must be i32
    // TODO update StageWriter to avoid all the useless casting.
    fn write<ES: Numeric, G: global::GlobalConfig>(
        this: &mut Self,
        mut slice: SliceMut<Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
        let tiling = config.stage_tiling(Ident::Out);

        let shape_col = tiling.tile_shape_col();

        let line_size = config.stage_line_size(Ident::Out);

        // TODO I assume row order here. Is this always true?
        let elem_per_unit = tiling.total_size() / (config.plane_dim() * line_size);

        for k in 0..elem_per_unit {
            let index = k + UNIT_POS_X * elem_per_unit;
            let row = index * line_size / shape_col;
            let col = index * line_size % shape_col;

            // All the ES::cast_from are trivial as I assume ES is i32.
            slice[index] -= Line::new(ES::cast_from(
                this.quantization.zero_offset_rhs * this.lhs_sums[row],
            ));
            slice[index] -= Line::new(ES::cast_from(
                this.quantization.zero_offset_lhs * this.rhs_sums[col],
            ));
            slice[index] += Line::new(ES::cast_from(
                this.quantization.zero_offset_lhs
                    * this.quantization.zero_offset_rhs
                    * i32::cast_from(config.shape().k),
            ));
            slice[index] = Line::cast_from(
                this.quantization
                    .scale_approx(Line::cast_from(slice[index])),
            );
            slice[index] += Line::new(ES::cast_from(this.quantization.zero_offset_rhs));

            slice[index] = clamp_to_u8_range(slice[index]);
        }

        TilewiseUnloading::unload_from_slice::<u8, ES, G>(
            &mut this.tensor_view,
            slice.to_slice(),
            compute_plane_offset,
            accumulator_offset,
            config,
        );
    }
}

#[cube]
fn clamp_to_u8_range<ES: Numeric>(mut x: Line<ES>) -> Line<ES> {
    let max = Line::new(ES::from_int(255));
    let min = Line::new(ES::from_int(0));
    x = select_many(x.greater_than(max), max, x);
    select_many(x.less_than(min), min, x)
}
