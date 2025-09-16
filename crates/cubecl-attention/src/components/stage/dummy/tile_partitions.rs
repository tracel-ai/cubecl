use std::cmp::max;
use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::global::dummy::QueryLoader;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig, tile::TileAttention};

#[derive(CubeType)]
pub struct Accumulators<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::Accumulator>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> Accumulators<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> Accumulators<AP, TA, S> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.seq_q * partition_size.val_dim) {
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
    ) -> &TA::Accumulator {
        let partition_size = config.tiling_scheme().partition_size;
        self.sequence
            .index(comptime!(i * partition_size.val_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut TA::Accumulator {
        let partition_size = config.tiling_scheme().partition_size;
        self.sequence
            .index_mut(comptime!(i * partition_size.val_dim + j))
    }
}

#[derive(CubeType)]
pub struct Queries<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::Query>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> Queries<AP, TA, S>
{
    pub fn new(query_loader: QueryLoader<AP>, #[comptime] config: S) -> Queries<AP, TA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        let mut i = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime!(p.seq_q) {
            let mut j = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime!(p.head_dim) {
                let tile = query_loader.get_tile::<S>(i, j, config);
                sequence.push(TA::init_query(&tile, config.tile_config()));

                comptime![j += 1];
            }

            comptime![i += 1];
        }

        Queries::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &TA::Query {
        let partition_size = config.tiling_scheme().partition_size;
        self.sequence
            .index(comptime!(i * partition_size.head_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut TA::Query {
        let partition_size = config.tiling_scheme().partition_size;
        self.sequence
            .index_mut(comptime!(i * partition_size.head_dim + j))
    }
}

#[derive(CubeType)]
pub struct KeyValues<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::KeyValue>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> KeyValues<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> KeyValues<AP, TA, S> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(max(partition_size.head_dim, partition_size.val_dim)) {
            sequence.push(TA::init_key_value(config.tile_config()));
        }

        KeyValues::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(&self, #[comptime] i: u32, #[comptime] _config: S) -> &TA::KeyValue {
        self.sequence.index(i)
    }

    pub fn get_at_mut(&mut self, #[comptime] i: u32, #[comptime] _config: S) -> &mut TA::KeyValue {
        self.sequence.index_mut(i)
    }
}

#[derive(CubeType)]
pub struct Scores<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> {
    sequence: Sequence<TA::ScoreProb>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<AP>,
    S: StageAttentionConfig<FlashMatmulConfig = TA::Config>,
> Scores<AP, TA, S>
{
    pub fn new(#[comptime] config: S) -> Scores<AP, TA, S> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.seq_q) {
            sequence.push(TA::init_score(config.tile_config()));
        }

        Scores::<AP, TA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(&self, #[comptime] i: u32) -> &TA::ScoreProb {
        self.sequence.index(i)
    }

    pub fn get_at_mut(&mut self, #[comptime] i: u32) -> &mut TA::ScoreProb {
        self.sequence.index_mut(i)
    }
}
