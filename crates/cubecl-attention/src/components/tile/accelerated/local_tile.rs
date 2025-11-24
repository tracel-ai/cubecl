use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::tile::{
    FragmentAccumulator, FragmentAccumulatorExpand, FragmentMask, FragmentMaskExpand, RowVal,
    RowWise, RowwiseFormat, RowwiseFormatExpand,
};

use crate::components::tile::{FragmentLayout, FragmentLayoutExpand};

#[derive(CubeType)]
/// Assumes:
/// - unit_size * plane_dim = total_size (not dim wise but in total count)
pub struct LocalTile<E: Numeric> {
    array: Array<E>,
    layout: LocalTileLayout,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum InnerLayout {
    /// Each unit has all its elements contiguous inside the same row
    ///
    ///  0,  0,  1,  1,  2,  2,  3,  3,
    ///  4,  4,  5,  5,  6,  6,  7,  7,
    ///  8,  8,  9,  9, 10, 10, 11, 11,
    /// 12, 12, 13, 13, 14, 14, 15, 15,
    /// 16, 16, 17, 17, 18, 18, 19, 19,
    /// 20, 20, 21, 21, 22, 22, 23, 23,
    /// 24, 24, 25, 25, 26, 26, 27, 27,
    /// 28, 28, 29, 29, 30, 30, 31, 31,
    Contiguous,
    /// Each unit spreads its elements along two rows
    ///
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    SplitRows,
}

#[cube]
impl<E: Numeric> LocalTile<E> {
    pub fn new(layout: LocalTileLayout) -> LocalTile<E> {
        let array = Array::<E>::new(comptime!(layout.unit_size.0 * layout.unit_size.1));
        LocalTile::<E> { array, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.unit_size.0 * self.layout.unit_size.1 {
            self.array[i] = E::from_int(0);
        }
    }

    pub fn fill_from_slice(&mut self, smem_slice: &Slice<E>) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = self.layout.absolute_pos((r, c));
                let index = row * self.layout.total_size.1 + col;

                self.array[r * self.layout.unit_size.1 + c] = smem_slice[index];
            }
        }
    }

    pub fn fill_from_strided_tile<E2: Numeric>(&mut self, strided_tile: &StridedTile<E2>) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = self.layout.absolute_pos((r, c));
                self.array[r * self.layout.unit_size.1 + c] =
                    E::cast_from(strided_tile.get_line(row, col))
            }
        }
    }

    pub fn store_to(&self, smem_slice: &mut SliceMut<E>) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = self.layout.absolute_pos((r, c));
                let index = row * self.layout.total_size.1 + col;

                smem_slice[index] = self.array[r * self.layout.unit_size.1 + c];
            }
        }
    }
}

#[derive(CubeType, Copy, Clone)]
pub struct LocalTileLayout {
    #[cube(comptime)]
    total_size: Coords2d,
    #[cube(comptime)]
    unit_size: Coords2d,
    #[cube(comptime)]
    num_units_per_row: u32,
    #[cube(comptime)]
    plane_dim: u32,
}

#[cube]
impl LocalTileLayout {
    pub fn new(
        #[comptime] total_size: Coords2d,
        #[comptime] plane_dim: u32,
        #[comptime] inner_layout: InnerLayout,
    ) -> LocalTileLayout {
        let total_elements = total_size.0 * total_size.1;
        let elements_per_unit = total_elements.div_ceil(plane_dim);

        let (num_rows_per_unit, num_cols_per_unit) = match inner_layout {
            InnerLayout::Contiguous => (1u32, elements_per_unit),
            InnerLayout::SplitRows => (2u32, elements_per_unit / 2u32),
        };
        let unit_size = (num_rows_per_unit, num_cols_per_unit);

        let num_units_per_row = comptime!(total_size.1 / unit_size.1);

        LocalTileLayout {
            total_size,
            unit_size,
            num_units_per_row,
            plane_dim,
        }
    }
}

#[cube]
impl FragmentLayout for LocalTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        let abs_row_index = {
            let row_0 = UNIT_POS_X / self.num_units_per_row;
            let row_jump = comptime!(self.plane_dim / self.num_units_per_row);

            local_pos.0 * row_jump + row_0
        };

        let abs_col_index = self.unit_size.1 * (UNIT_POS_X % self.num_units_per_row) + local_pos.1;

        (abs_row_index, abs_col_index)
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(self.total_size.1 / self.unit_size.1)
    }
}

#[cube]
impl<E: Float> RowwiseFormat<E> for LocalTile<E> {
    type Layout = LocalTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            let mut val = E::min_value();

            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                val = Max::max(val, self.array[index]);
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.unit_size.0,
            vals,
        }
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            let mut val = E::from_int(0);

            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                val += self.array[index];
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.unit_size.0,
            vals,
        }
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        #[unroll]
        for r in 0..this.layout.unit_size.0 {
            let row_offset = r * this.layout.unit_size.1;
            #[unroll]
            for c in 0..this.layout.unit_size.1 {
                let index = row_offset + c;
                this.array[index] = this.array[index] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    fn exp_diff(&mut self, val: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index] = Exp::exp(self.array[index] - val.index(r));
            }
        }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for LocalTile<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index] = self.array[index] * scale.index(r);
            }
        }
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for LocalTile<E> {
    type Layout = LocalTileLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.array[local_pos.0 * self.layout.unit_size.1 + local_pos.1])
    }
}
