use std::cmp::max;
use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::global::dummy::QueryReader;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig, tile::TileAttention};

#[derive(CubeType)]
pub struct Accumulators<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::AccumulatorTile>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> Accumulators<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> Accumulators<AP, TA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.val_dim) {
            sequence.push(TA::init_accumulator(config.tile_config()));
        }

        Accumulators::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &TA::AccumulatorTile {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(i * p.val_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut TA::AccumulatorTile {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(i * p.val_dim + j))
    }
}

#[derive(CubeType)]
pub struct Queries<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::QueryTile>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> Queries<AP, TA, S>
{
    pub fn new(query_loader: QueryReader<AP>, #[comptime] config: S) -> Queries<AP, TA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        let mut q = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime!(p.seq_q) {
            let mut hd = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime!(p.head_dim) {
                let tile = query_loader.get_tile::<S>((q, hd).runtime(), config);
                sequence.push(TA::init_query(&tile, config.tile_config()));

                comptime![hd += 1];
            }

            comptime![q += 1];
        }

        Queries::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &TA::QueryTile {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(q * p.head_dim + hd))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &mut TA::QueryTile {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(q * p.head_dim + hd))
    }
}

#[derive(CubeType)]
pub enum KeyValues<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> {
    Reuse(KeyValueSequence<AP, TA, S>),
    Separate(KeyValueSequence<AP, TA, S>, KeyValueSequence<AP, TA, S>),
}

#[derive(CubeType)]
pub struct KeyValueSequence<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::KeyValueTile>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> KeyValues<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> KeyValues<AP, TA, S> {
        if config.reuse_key_value() {
            let p = config.tiling_scheme().partition_size;
            let mut sequence = Sequence::new();

            #[unroll]
            for _ in 0..comptime!(p.seq_kv * max(p.head_dim, p.val_dim)) {
                sequence.push(TA::init_key_value(config.tile_config()));
            }

            KeyValues::<AP, TA, S>::new_Reuse(KeyValueSequence::<AP, TA, S> {
                sequence,
                _phantom: PhantomData,
            })
        } else {
            let p = config.tiling_scheme().partition_size;
            let mut keys = Sequence::new();
            let mut values = Sequence::new();

            #[unroll]
            for _ in 0..comptime!(p.head_dim * p.seq_kv) {
                keys.push(TA::init_key(config.tile_config()));
            }
            #[unroll]
            for _ in 0..comptime!(p.seq_kv * p.val_dim) {
                values.push(TA::init_value(config.tile_config()));
            }

            KeyValues::<AP, TA, S>::new_Separate(
                KeyValueSequence::<AP, TA, S> {
                    sequence: keys,
                    _phantom: PhantomData,
                },
                KeyValueSequence::<AP, TA, S> {
                    sequence: values,
                    _phantom: PhantomData,
                },
            )
        }
    }

    pub fn get_key_at(
        &self,
        #[comptime] hd: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &TA::KeyValueTile {
        let index = hd * config.tiling_scheme().partition_size.seq_kv + kv;
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index(index),
            KeyValues::Separate(keys, _) => keys.sequence.index(index),
        }
    }

    pub fn get_key_at_mut(
        &mut self,
        #[comptime] hd: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &mut TA::KeyValueTile {
        let index = hd * config.tiling_scheme().partition_size.seq_kv + kv;
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index_mut(index),
            KeyValues::Separate(keys, _) => keys.sequence.index_mut(index),
        }
    }

    pub fn get_value_at(
        &self,
        #[comptime] kv: u32,
        #[comptime] vd: u32,
        #[comptime] config: S,
    ) -> &TA::KeyValueTile {
        let index = kv * config.tiling_scheme().partition_size.val_dim + vd;
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index(index),
            KeyValues::Separate(_, values) => values.sequence.index(index),
        }
    }

    pub fn get_value_at_mut(
        &mut self,
        #[comptime] kv: u32,
        #[comptime] vd: u32,
        #[comptime] config: S,
    ) -> &mut TA::KeyValueTile {
        let index = kv * config.tiling_scheme().partition_size.val_dim + vd;
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index_mut(index),
            KeyValues::Separate(_, values) => values.sequence.index_mut(index),
        }
    }
}

#[derive(CubeType)]
pub struct SoftmaxPartition<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::SoftmaxTile>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = TA::Config>,
> SoftmaxPartition<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> SoftmaxPartition<AP, TA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.seq_kv) {
            sequence.push(TA::init_softmax(config.tile_config()));
        }

        SoftmaxPartition::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &TA::SoftmaxTile {
        let index = q * config.tiling_scheme().partition_size.seq_kv + kv;
        self.sequence.index(index)
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &mut TA::SoftmaxTile {
        let index = q * config.tiling_scheme().partition_size.seq_kv + kv;
        self.sequence.index_mut(index)
    }
}
