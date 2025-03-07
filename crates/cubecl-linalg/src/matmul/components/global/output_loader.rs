use super::args::Quantization;
use super::GlobalConfig;
use crate::matmul::components::{
    global::{self, tensor_view::TensorWriter, tilewise_unloading::TilewiseUnloading},
    stage::StageWriter,
    Ident,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

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

// TODO To make the type system happy, I use EA, but it should
//      always be i32.
#[derive(CubeType, Clone)]
pub struct Quantizer<EA: Numeric> {
    pub lhs_sums: SharedMemory<EA>, // TODO Use Line
    pub rhs_sums: SharedMemory<EA>,
    pub quantization: Quantization,
}

#[cube]
impl<EA: Numeric> Quantizer<EA> {
    pub fn new<G: GlobalConfig>(quantization: Quantization, #[comptime] config: G) -> Self {
        let row = config.tiling_dimensions(Ident::Lhs).total_row();
        let col = config.tiling_dimensions(Ident::Rhs).total_col();
        let unit_count = config.num_planes() * config.plane_dim();

        let mut lhs_sums = SharedMemory::new(row);
        let row_count_per_unit = row / unit_count;
        for k in 0..row_count_per_unit {
            lhs_sums[k + UNIT_POS * row_count_per_unit] = EA::from_int(0);
        }

        let mut rhs_sums = SharedMemory::new(col);
        let col_count_per_unit = col / unit_count;
        for k in 0..col_count_per_unit {
            rhs_sums[k + UNIT_POS * row_count_per_unit] = EA::from_int(0);
        }

        Quantizer::<EA> {
            lhs_sums,
            rhs_sums,
            quantization,
        }
    }

    pub fn add_quantization_into<G: global::GlobalConfig>(
        &self,
        slice: &mut SliceMut<Line<EA>>,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(Ident::Out);

        let shape_col = tiling.tile_shape_col();

        let line_size = config.stage_line_size(Ident::Out);

        // TODO I assume row order here. Is this always true?
        let elem_per_unit = tiling.total_size() / (config.plane_dim() * line_size);

        for k in 0..elem_per_unit {
            let index = k + UNIT_POS_X * elem_per_unit;
            let row = index * line_size / shape_col;
            let col = index * line_size % shape_col;

            // All the EA::cast_from are trivial as I assume EA is i32.
            slice[index] -=
                Line::new(EA::cast_from(self.quantization.zero_offset_rhs) * self.lhs_sums[row]);
            slice[index] -=
                Line::new(EA::cast_from(self.quantization.zero_offset_lhs) * self.rhs_sums[col]);
            slice[index] += Line::new(EA::cast_from(
                self.quantization.zero_offset_lhs
                    * self.quantization.zero_offset_rhs
                    * self.quantization.shape_k,
            ));
            slice[index] = Line::cast_from(
                self.quantization
                    .scale_approx(Line::cast_from(slice[index])),
            );
            slice[index] += Line::new(EA::cast_from(self.quantization.zero_offset_rhs));

            slice[index] = clamp_to_u8_range(slice[index]);
        }
    }
}

#[cube]
fn clamp_to_u8_range<ES: Numeric>(mut x: Line<ES>) -> Line<ES> {
    let max = Line::new(ES::from_int(255));
    let min = Line::new(ES::from_int(0));
    x = select_many(x.greater_than(max), max, x);
    select_many(x.less_than(min), min, x)
}
