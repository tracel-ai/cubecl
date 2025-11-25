use std::cmp::max;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::RowVal;
use crate::components::tile::RowWise;
use crate::components::tile::unit_register::setup::UnitTileAttentionConfig;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::{RowwiseFormat, RowwiseFormatExpand};

use crate::components::tile::TileAttention;
use crate::components::tile::{FragmentLayout, FragmentLayoutExpand};

pub struct UnitRegisterTileAttention;

#[derive(CubeType)]
pub struct UnitTile<E: Numeric> {
    data: Array<E>,
    layout: UnitTileLayout,
}

#[derive(CubeType, Copy, Clone)]
pub struct UnitTileLayout {
    #[cube(comptime)]
    num_rows: u32,
    #[cube(comptime)]
    num_cols: u32,
}

#[cube]
impl<E: Numeric> UnitTile<E> {
    pub fn new(layout: UnitTileLayout) -> UnitTile<E> {
        let data = Array::<E>::new(comptime!(layout.num_rows * layout.num_cols));
        UnitTile::<E> { data, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.num_rows * self.layout.num_cols {
            self.data[i] = E::from_int(0);
        }
    }

    pub fn get(&self, i: u32, j: u32) -> E {
        self.data[i * self.layout.num_cols + j]
    }

    pub fn accumulate(&mut self, i: u32, j: u32, val: E) {
        self.data[i * self.layout.num_cols + j] += val;
    }
}

#[cube]
impl UnitTileLayout {
    pub fn new(#[comptime] num_rows: u32, #[comptime] num_cols: u32) -> UnitTileLayout {
        UnitTileLayout { num_rows, num_cols }
    }
}

#[cube]
impl FragmentLayout for UnitTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        local_pos
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        1u32
    }
}

#[cube]
impl<E: Float> RowwiseFormat<E> for UnitTile<E> {
    type Layout = UnitTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::min_value();

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val = Max::max(val, self.data[index]);
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows,
            vals,
        }
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::from_int(0);

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val += self.data[index];
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows,
            vals,
        }
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        #[unroll]
        for r in 0..this.layout.num_rows {
            let row_offset = r * this.layout.num_cols;
            #[unroll]
            for c in 0..this.layout.num_cols {
                let index = row_offset + c;
                this.data[index] = this.data[index] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    fn exp_diff(&mut self, val: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index] = Exp::exp(self.data[index] - val.index(r));
            }
        }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for UnitTile<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index] = self.data[index] * scale.index(r);
            }
        }
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for UnitTile<E> {
    type Layout = UnitTileLayout;
    type SoftmaxScore = UnitTile<E>;
    type SoftmaxRowFormat = UnitTile<E>;
    type SoftmaxVal = UnitTile<E>;

    fn rowwise_mut(&mut self) -> &mut UnitTile<E> {
        self
    }

    fn update_from_rowwise(&mut self) {
        // Nothing to do, because rowwise = self
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for UnitTile<E> {
    type Layout = UnitTileLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.data[local_pos.0 * self.layout.num_cols + local_pos.1])
    }
}

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for UnitRegisterTileAttention {
    type Config = UnitTileAttentionConfig;

    type Query = UnitTile<QT<AP>>;
    type KeyValue = UnitTile<KVT<AP>>;
    type Mask = UnitTile<MSK<AP>>;
    type Softmax = UnitTile<SM<AP>>;
    type SoftmaxRow = UnitTile<SM<AP>>;
    type Accumulator = UnitTile<ACC<AP>>;
    type FragmentLayout = UnitTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout {
        UnitTileLayout {
            num_rows: config.shared.attention_tile_size.seq_q,
            num_cols: config.shared.attention_tile_size.seq_kv,
        }
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.shared.attention_tile_size.to_score_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.shared.attention_tile_size.to_value_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            comptime!(max(
                config.shared.attention_tile_size.head_dim,
                config.shared.attention_tile_size.seq_kv,
            )),
            comptime!(max(
                config.shared.attention_tile_size.seq_kv,
                config.shared.attention_tile_size.val_dim,
            )),
        ))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.head_dim,
            config.shared.attention_tile_size.seq_kv,
        ))
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_kv,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        UnitTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        UnitTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.head_dim,
        ))
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_transposed_unit_tile(tile, fragment);
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] _config: Self::Config,
    ) {
        array_tile_to_slice(out, slice)
    }
}

#[cube]
fn strided_tile_to_unit_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    unit_tile: &mut UnitTile<E2>,
) {
    let line_size = strided_tile.line_size;
    assert!(unit_tile.layout.num_cols % line_size == 0);

    let col_iterations = comptime!(unit_tile.layout.num_cols / strided_tile.line_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let line_read = strided_tile.get_line(row, col);
            #[unroll]
            for i in 0..line_size {
                unit_tile.data[row * unit_tile.layout.num_cols + col * line_size + i] =
                    E2::cast_from(line_read[i]);
            }
        }
    }
}

#[cube]
fn strided_tile_to_transposed_unit_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    unit_tile: &mut UnitTile<E2>,
) {
    for row in 0..unit_tile.layout.num_rows {
        for col in 0..unit_tile.layout.num_cols {
            unit_tile.data[row * unit_tile.layout.num_cols + col] =
                E2::cast_from(strided_tile.get_line(col, row))
        }
    }
}

#[cube]
fn array_tile_to_slice<E: Numeric, E2: Numeric>(
    unit_tile: &UnitTile<E>,
    slice: &mut SliceMut<Line<E2>>,
) {
    for row in 0..unit_tile.layout.num_rows {
        for col in 0..unit_tile.layout.num_cols {
            let index = row * unit_tile.layout.num_cols + col;
            slice[index] = Line::cast_from(unit_tile.data[index]);
        }
    }
}

#[cube]
fn unit_inner_matmul<Lhs: Float, Rhs: Float, Acc: Float>(
    lhs: &UnitTile<Lhs>,
    rhs: &UnitTile<Rhs>,
    out: &mut UnitTile<Acc>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    for m_ in 0..m {
        for n_ in 0..n {
            let mut sum = Acc::from_int(0);
            for k_ in 0..k {
                let lhs_val = lhs.get(m_, k_);
                let rhs_val = rhs.get(k_, n_);
                sum += Acc::cast_from(lhs_val) * Acc::cast_from(rhs_val);
            }
            out.accumulate(m_, n_, sum);
        }
    }
}
