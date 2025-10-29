use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::unit_register::UnitRegisterFragmentAttentionConfig;
use crate::components::fragment::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::RowWise;

use crate::components::fragment::FragmentAttention;
use crate::components::fragment::{FragmentLayout, FragmentLayoutExpand};
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};

pub struct UnitRegisterFragmentAttention;

#[derive(CubeType)]
pub struct UnitTile<E: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<E>,
}

#[derive(CubeType)]
pub struct UnitTileLayout {}

#[cube]
impl FragmentLayout for UnitTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        todo!()
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        todo!()
    }
}

#[cube]
impl<E: Float> FragmentOps<E> for UnitTile<E> {
    type Layout = UnitTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, val: &RowWise<E>) {
        todo!()
    }

    fn layout(&self) -> Self::Layout {
        todo!()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for UnitTile<E> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        todo!()
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
        todo!()
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        todo!()
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        todo!()
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        todo!()
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        todo!()
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        todo!()
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        todo!()
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        todo!()
    }

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {}

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] _config: Self::Config) {}

    fn zero_accumulator(acc: &mut Self::Accumulator) {}

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] _config: Self::Config,
    ) {
    }
}
