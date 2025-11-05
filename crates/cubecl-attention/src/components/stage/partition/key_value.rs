use std::cmp::max;
use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::fragment::FragmentAttention;
use crate::components::tile::KeyValueTile;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// To save registers, key and value can reuse the same fragment, since they run sequentially.
/// Or they can be separate if desired or if shapes mismatch.
///
/// For each `kv`:
/// - Key: iterate over one column of `head_dim`, multiplying each (hd, kv) tile with all `seq_q` tiles.
/// - Value: then iterate over one row of `val_dim`, multiplying each (kv, vd) tile with all `seq_q` tiles.
///
/// Only one tile is active at a time; key and value alternate per `kv`.
pub enum KeyValues<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> {
    Reuse(KeyValueSequence<AP, FA, S>),
    Separate(KeyValueSequence<AP, FA, S>, KeyValueSequence<AP, FA, S>),
}

#[derive(CubeType)]
pub struct KeyValueSequence<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> {
    sequence: Sequence<KeyValueTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> KeyValues<AP, FA, S>
{
    pub fn new(#[comptime] config: S) -> KeyValues<AP, FA, S> {
        if config.reuse_key_value() {
            let p = config.tiling_scheme().partition_size;
            let mut sequence = Sequence::new();

            #[unroll]
            for _ in 0..comptime!(p.seq_kv * max(p.head_dim, p.val_dim)) {
                sequence.push(KeyValueTile::new_key_value(config.tile_config()));
            }

            KeyValues::<AP, FA, S>::new_Reuse(KeyValueSequence::<AP, FA, S> {
                sequence,
                _phantom: PhantomData,
            })
        } else {
            let mut keys = Sequence::new();
            let mut values = Sequence::new();

            keys.push(KeyValueTile::new_key(config.tile_config()));
            values.push(KeyValueTile::new_value(config.tile_config()));

            KeyValues::<AP, FA, S>::new_Separate(
                KeyValueSequence::<AP, FA, S> {
                    sequence: keys,
                    _phantom: PhantomData,
                },
                KeyValueSequence::<AP, FA, S> {
                    sequence: values,
                    _phantom: PhantomData,
                },
            )
        }
    }

    pub fn get_key(&self) -> &KeyValueTile<AP, FA> {
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index(0u32),
            KeyValues::Separate(keys, _) => keys.sequence.index(0u32),
        }
    }

    pub fn get_key_mut(&mut self) -> &mut KeyValueTile<AP, FA> {
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index_mut(0u32),
            KeyValues::Separate(keys, _) => keys.sequence.index_mut(0u32),
        }
    }

    pub fn get_value(&self) -> &KeyValueTile<AP, FA> {
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index(0u32),
            KeyValues::Separate(_, values) => values.sequence.index(0u32),
        }
    }

    pub fn get_value_mut(&mut self) -> &mut KeyValueTile<AP, FA> {
        match self {
            KeyValues::Reuse(key_values) => key_values.sequence.index_mut(0u32),
            KeyValues::Separate(_, values) => values.sequence.index_mut(0u32),
        }
    }
}
