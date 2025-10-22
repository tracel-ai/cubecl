use std::cmp::max;
use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::AttentionTilingScheme;
use crate::components::fragment::AttentionMatmul;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::KeyValueTile;
use crate::components::tile::MaskTile;
use crate::components::tile::QueryTile;
use crate::components::tile::SoftmaxTile;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig, tile::TileAttention};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
pub struct QueryPartition<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    sequence: Sequence<QueryTile<AP, AM>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> QueryPartition<AP, AM, S>
{
    pub fn new(#[comptime] config: S) -> QueryPartition<AP, AM, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.head_dim) {
            sequence.push(TileAttention::init_query(config.tile_config()));
        }

        QueryPartition::<AP, AM, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &QueryTile<AP, AM> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(q * p.head_dim + hd))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &mut QueryTile<AP, AM> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(q * p.head_dim + hd))
    }
}

#[derive(CubeType)]
pub enum KeyValues<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    Reuse(KeyValueSequence<AP, AM, S>),
    Separate(KeyValueSequence<AP, AM, S>, KeyValueSequence<AP, AM, S>),
}

#[derive(CubeType)]
pub struct KeyValueSequence<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    sequence: Sequence<KeyValueTile<AP, AM>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> KeyValues<AP, AM, S>
{
    pub fn new(#[comptime] config: S) -> KeyValues<AP, AM, S> {
        if config.reuse_key_value() {
            let p = config.tiling_scheme().partition_size;
            let mut sequence = Sequence::new();

            #[unroll]
            for _ in 0..comptime!(p.seq_kv * max(p.head_dim, p.val_dim)) {
                sequence.push(KeyValueTile::new_key_value(config.tile_config()));
            }

            KeyValues::<AP, AM, S>::new_Reuse(KeyValueSequence::<AP, AM, S> {
                sequence,
                _phantom: PhantomData,
            })
        } else {
            let p = config.tiling_scheme().partition_size;
            let mut keys = Sequence::new();
            let mut values = Sequence::new();

            #[unroll]
            for _ in 0..comptime!(p.head_dim * p.seq_kv) {
                keys.push(KeyValueTile::new_key(config.tile_config()));
            }
            #[unroll]
            for _ in 0..comptime!(p.seq_kv * p.val_dim) {
                values.push(KeyValueTile::new_value(config.tile_config()));
            }

            KeyValues::<AP, AM, S>::new_Separate(
                KeyValueSequence::<AP, AM, S> {
                    sequence: keys,
                    _phantom: PhantomData,
                },
                KeyValueSequence::<AP, AM, S> {
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
    ) -> &KeyValueTile<AP, AM> {
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
    ) -> &mut KeyValueTile<AP, AM> {
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
    ) -> &KeyValueTile<AP, AM> {
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
    ) -> &mut KeyValueTile<AP, AM> {
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
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    sequence: Sequence<SoftmaxTile<AP, AM>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> SoftmaxPartition<AP, AM, S>
{
    pub fn new(#[comptime] config: S) -> SoftmaxPartition<AP, AM, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.seq_kv) {
            sequence.push(SoftmaxTile::new(config.tile_config()));
        }

        SoftmaxPartition::<AP, AM, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &SoftmaxTile<AP, AM> {
        let index = q * config.tiling_scheme().partition_size.seq_kv + kv;
        self.sequence.index(index)
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] config: S,
    ) -> &mut SoftmaxTile<AP, AM> {
        let index = q * config.tiling_scheme().partition_size.seq_kv + kv;
        self.sequence.index_mut(index)
    }
}

#[derive(CubeType)]
pub struct MaskPartition<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    sequence: Sequence<MaskTile<AP, AM>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> MaskPartition<AP, AM, S>
{
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: S,
    ) -> MaskPartition<AP, AM, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        let mut q = comptime![0];

        #[unroll]
        for _ in 0..p.seq_q {
            let mut kv = comptime![0];

            #[unroll]
            for _ in 0..p.seq_kv {
                sequence.push(MaskTile::new(out_of_bounds, (q, kv), config.tile_config()));

                comptime![kv += 1];
            }

            comptime![q += 1];
        }

        MaskPartition::<AP, AM, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> &MaskTile<AP, AM> {
        let p = tiling_scheme.partition_size;
        self.sequence.index(comptime!(q * p.seq_kv + kv))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> &mut MaskTile<AP, AM> {
        let p = tiling_scheme.partition_size;
        self.sequence.index_mut(comptime!(q * p.seq_kv + kv))
    }
}

#[derive(CubeType)]
pub struct AccumulatorPartition<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> {
    sequence: Sequence<AccumulatorTile<AP, AM>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    AM: AttentionMatmul<AP>,
    S: StageAttentionConfig<AttentionMatmulConfig = AM::Config>,
> AccumulatorPartition<AP, AM, S>
{
    pub fn new(#[comptime] config: S) -> AccumulatorPartition<AP, AM, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.val_dim) {
            sequence.push(AccumulatorTile::new(config.tile_config()));
        }

        AccumulatorPartition::<AP, AM, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &AccumulatorTile<AP, AM> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(i * p.val_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut AccumulatorTile<AP, AM> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(i * p.val_dim + j))
    }
}
