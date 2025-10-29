use std::cmp::max;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::FragmentAttentionConfig;
use crate::components::fragment::unit_register::UnitRegisterFragmentAttentionConfig;
use crate::components::fragment::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::RowVal;
use crate::components::tile::RowWise;

use crate::components::fragment::FragmentAttention;
use crate::components::fragment::{FragmentLayout, FragmentLayoutExpand};
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};

pub struct UnitRegisterFragmentAttention;

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
impl<E: Float> FragmentOps<E> for UnitTile<E> {
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

    fn layout(&self) -> Self::Layout {
        self.layout
    }
}

#[cube]
impl<E: Numeric> FragmentMask for UnitTile<E> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.data[local_pos.0 * self.layout.num_cols + local_pos.1])
    }
}

#[cube]
impl<AP: AttentionPrecision> FragmentAttention<AP> for UnitRegisterFragmentAttention {
    type Config = UnitRegisterFragmentAttentionConfig;

    type Query = UnitTile<QT<AP>>;
    type KeyValue = UnitTile<KVT<AP>>;
    type Mask = UnitTile<MSK<AP>>;
    type Softmax = UnitTile<SM<AP>>;
    type Accumulator = UnitTile<ACC<AP>>;
    type FragmentLayout = UnitTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout {
        UnitTileLayout {
            num_rows: config.attention_tile_size().seq_q,
            num_cols: config.attention_tile_size().seq_kv,
        }
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_score_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_value_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            comptime!(max(
                config.attention_tile_size().head_dim,
                config.attention_tile_size().seq_kv,
            )),
            comptime!(max(
                config.attention_tile_size().seq_kv,
                config.attention_tile_size().val_dim,
            )),
        ))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.attention_tile_size().head_dim,
            config.attention_tile_size().seq_kv,
        ))
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.attention_tile_size().seq_kv,
            config.attention_tile_size().val_dim,
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        UnitTile::new(<Self as FragmentAttention<AP>>::softmax_layout(config))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        UnitTile::new(<Self as FragmentAttention<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        UnitTile::new(UnitTileLayout::new(
            config.attention_tile_size().seq_q,
            config.attention_tile_size().val_dim,
        ))
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        UnitTile::new(UnitTileLayout::new(
            config.attention_tile_size().seq_q,
            config.attention_tile_size().head_dim,
        ))
    }

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        strided_tile_to_array_tile(tile, fragment);
    }

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, fragment);
    }

    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, fragment);
    }

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] _config: Self::Config) {
        softmax.zero();
    }

    fn zero_accumulator(acc: &mut Self::Accumulator) {
        acc.zero();
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
fn strided_tile_to_array_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    unit_tile: &mut UnitTile<E2>,
) {
    for row in 0..unit_tile.layout.num_rows {
        for col in 0..unit_tile.layout.num_cols {
            unit_tile.data[row * unit_tile.layout.num_cols + col] =
                E2::cast_from(strided_tile.get_line(row, col))
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
