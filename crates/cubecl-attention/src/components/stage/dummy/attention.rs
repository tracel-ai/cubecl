use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{stage::StageToTileReader, tile::TileMatmul};
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    global::dummy::{DummyAccumulator, DummyWriter, GlobalToTileReader},
    stage::{StageAttention, StageConfig as _, dummy::config::DummyStageConfig},
};

pub struct DummyStageAttention<
    AP: AttentionPrecision,
    STM: TileMatmul<AP::MatmulPrecision>,
    VTM: TileMatmul<AP::MatmulPrecision>,
    R,
> {
    _phantom: PhantomData<(AP, STM, VTM, R)>,
}

#[cube]
impl<
    STM: TileMatmul<AP::MatmulPrecision>,
    VTM: TileMatmul<AP::MatmulPrecision>,
    AP: AttentionPrecision,
    R: StageToTileReader<AP::ES>,
> StageAttention<AP> for DummyStageAttention<AP, STM, VTM, R>
{
    type Config = DummyStageConfig<STM::Config, VTM::Config>;

    type KeyReader = R;
    type ValueReader = R;
    type Accumulator = DummyAccumulator;
    type Writer = DummyWriter;

    type State = DummyStageState<AP::EI>;

    // Tc times, each call is at an index j
    // Return (m_ij, l_ij) [new]
    fn execute(
        query_reader: &GlobalToTileReader,
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        acc: &mut Self::Accumulator,
        prev_state: &Self::State,
        #[comptime] config: Self::Config,
    ) -> Self::State {
        // Not accurate: this does two partition_matmul with computation in between,
        // But there should be a partition attention that computes intermediary stuff at each tile.
        //
        // Go in the partition i for unit/plane i, then do:
        // -> Not exactly the algo's i, this is the block id, now we're sub block, per unit/plane
        // S_ij = partition_matmul(query_reader, key_reader, acc=0) [reuse from matmul], acc is always resetted to 0
        // let rms = rowmax(S_ij)
        // m_ij = max(prev_m, rms)
        // sminusm = S_ij - m_ij <- broadcast the vector m_ij which has as many elems as S_ij has rows
        // P_ij = exp_elementwise(sminusm)
        // let rsp = rowsum(P_ij)
        // let epmm = e^(prev_m - mij)
        // l_ij = e^(prev_m - mij) * prev_l + rsp
        // Manually scale acc inplace:
        // acc = 1/diag(epmm) * acc
        // partition_matmul(P_ij, value_reader, acc=acc (scaled))
        // (m_ij, l_ij)

        // tmp
        DummyStageState::<AP::EI> {
            prev_m: Array::new(config.rows_per_plane()),
            prev_l: Array::new(config.rows_per_plane()),
        }
    }

    fn last_update(acc: &mut Self::Accumulator, prev_state: Self::State) {
        // O_i = 1/diag(l_i_Tc) x O_i_Tc
        todo!()
    }

    fn init_state() -> Self::State {
        todo!()
    }

    fn write(acc: &Self::Accumulator, writer: Self::Writer) {
        todo!()
    }

    fn zero_accumulator(acc: &mut Self::Accumulator) {
        todo!()
    }
}

#[derive(CubeType)]
pub struct DummyStageState<E: Float> {
    // Equal m_i'(j-1)
    prev_m: Array<E>,
    prev_l: Array<E>,
}
