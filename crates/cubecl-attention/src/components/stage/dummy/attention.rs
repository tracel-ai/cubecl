use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{MatrixLayout, stage::StageToTileReader, tile::Tile};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::tile::{ScoreMatmul, ValueMatmul};
use crate::components::{
    AttentionPrecision,
    global::GlobalAttentionConfig,
    stage::{
        StageAttention, StageAttentionConfig,
        dummy::{
            AccumulatorFragment, AttentionStageMemoryConfig, DummyWriter, KeyValueFragment,
            QueryFragment, ScoreProbFragment,
        },
    },
};

pub struct DummyStageAttention<
    AP: AttentionPrecision,
    SM: ScoreMatmul<AP>,
    VM: ValueMatmul<AP>,
    R,
    S: StageAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>,
> {
    _phantom: PhantomData<(AP, SM, VM, R, S)>,
}

#[cube]
impl<
    SM: ScoreMatmul<AP>,
    VM: ValueMatmul<AP>,
    AP: AttentionPrecision,
    R: StageToTileReader<AP::ES>,
    S: StageAttentionConfig<
            ScoreConfig = SM::Config,
            ValueConfig = VM::Config,
            ScoreStageMemoryConfig = AttentionStageMemoryConfig<SM::Config>,
            ValueStageMemoryConfig = AttentionStageMemoryConfig<VM::Config>,
        >,
> StageAttention<AP> for DummyStageAttention<AP, SM, VM, R, S>
{
    type Config = S;

    type KeyReader = R;
    type ValueReader = R;
    type Writer = DummyWriter<AP::EO>;

    type State = DummyStageState<AP::EA>;

    type Query = QueryFragment<AP, SM>;
    type KeyValue = KeyValueFragment<AP, SM, VM>;
    type ScoreProb = ScoreProbFragment<AP, SM, VM>;
    type Accumulator = AccumulatorFragment<AP, VM>;

    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    ) {
        comment!("Stage: Execute");
        // 1/sqrt(8)
        let inv_sqrt_dk = AP::EA::new(0.35355339059);

        let prev_m = state.m;
        let prev_l = state.l;
        let row = UNIT_POS_X / 4;
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;
        let mut tmp_smem = SharedMemory::<AP::EA>::new(64);

        comment!("Stage-Execute: Put K in fragment from reader for Score Matmul");
        let key_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
            AttentionStageMemoryConfig<SM::Config>,
        >(key_reader, 0, 0, config.score_stage_memory_config());
        SM::fill_rhs(&key_tile, key_value.key_mut(), config.score_config());

        comment!("Stage-Execute: Score matmul S=Q·K+0");
        SM::execute(
            &query.fragment,
            key_value.key(),
            score_prob.score_mut(),
            config.score_config(),
        );

        comment!(
            "Stage-Execute: Make sure we work with the right registers for scores, and scale them"
        );
        // TODO work on scores register directly
        SM::write_results::<AP::EA>(
            &mut score_prob.score(),
            &mut tmp_smem.to_slice_mut().try_cast_unchecked(),
            config.score_config(),
        );
        tmp_smem[index_0] *= inv_sqrt_dk;
        tmp_smem[index_1] *= inv_sqrt_dk;

        comment!("Stage-Execute: Compute running m");
        let mut m = prev_m;
        for i in 0..8 {
            let ts = tmp_smem[row * 8 + i];
            if ts > m {
                m = ts;
            }
        }

        comment!("Stage-Execute: Compute P and put it to fragment for Value Matmul");
        tmp_smem[index_0] = Exp::exp(tmp_smem[index_0] - m);
        tmp_smem[index_1] = Exp::exp(tmp_smem[index_1] - m);
        let p = Tile::<AP::ES> {
            slice: tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };
        VM::fill_lhs(&p, score_prob.prob_mut(), config.value_config());

        comment!("Stage-Execute: Compute running l");
        let epm = Exp::exp(prev_m - m);
        let mut rowsum = AP::EA::from_int(0);
        for i in 0..8 {
            rowsum += tmp_smem[row * 8 + i];
        }
        let l = epm * prev_l + rowsum;

        comment!("Stage-Execute: Put V in fragment from reader for Value Matmul");
        let value_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
            AttentionStageMemoryConfig<VM::Config>,
        >(value_reader, 0, 0, config.value_stage_memory_config());
        VM::fill_rhs(&value_tile, key_value.value_mut(), config.value_config());

        comment!("Stage-Execute: Scale acc by epm");
        // TODO modify registers directly when we are certain we are in the right row
        // Instead of storing modifying then refilling
        VM::write_results::<AP::EA>(
            &accumulator.fragment,
            &mut tmp_smem.to_slice_mut().try_cast_unchecked(),
            config.value_config(),
        );
        tmp_smem[index_0] *= epm;
        tmp_smem[index_1] *= epm;
        let tile = Tile::<AP::EA> {
            slice: tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };
        VM::fill_accumulator(&tile, &mut accumulator.fragment, config.value_config());

        comment!("Stage-Execute: Value Matmul O = P·V + scaled_O");
        VM::execute(
            score_prob.prob(),
            &key_value.value(),
            &mut accumulator.fragment,
            config.value_config(),
        );

        state.m = m;
        state.l = l;
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] config: Self::Config) {
        comment!("Stage: Rescale");
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;
        let mut tmp_smem = SharedMemory::<AP::EA>::new(64);

        VM::write_results::<AP::EA>(
            &acc.fragment,
            &mut tmp_smem.to_slice_mut().try_cast_unchecked(),
            config.value_config(),
        );

        tmp_smem[index_0] /= state.l;
        tmp_smem[index_1] /= state.l;
        let tile = Tile::<AP::EA> {
            slice: tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };
        VM::fill_accumulator(&tile, &mut acc.fragment, config.value_config());
    }

    fn init_state(#[comptime] _config: Self::Config) -> Self::State {
        comment!("Stage: Init Stage");

        DummyStageState::<AP::EA> {
            // TODO Neg infinity
            m: AP::EA::from_int(-99999999999),
            l: AP::EA::from_int(0),
        }
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        comment!("Stage: Write");
        let mut out_smem = SharedMemory::<AP::EA>::new(64);
        VM::write_results::<AP::EA>(
            &acc.fragment,
            &mut out_smem.to_slice_mut().try_cast_unchecked(),
            stage_config.value_config(),
        );

        DummyWriter::<AP::EO>::write::<G>(
            writer,
            out_smem.to_slice().try_cast_unchecked(),
            0,
            0,
            global_config,
        )
    }

    fn init_writer(out: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer {
        DummyWriter::new(out, 0, 0, 0)
    }

    fn init_fragments(
        query_reader: QueryRegisterReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (
        Self::Query,
        Self::KeyValue,
        Self::ScoreProb,
        Self::Accumulator,
    ) {
        (
            Self::Query::new(query_reader, config.score_config()),
            Self::KeyValue::new::<Self::Config>(config),
            Self::ScoreProb::new::<Self::Config>(config),
            Self::Accumulator::new(config.value_config()),
        )
    }
}

#[derive(CubeType)]
// There should be two strategies for state
// - Elect: one thread holds the state and shares it with row neighbours when necessary (needs broadcast at the beginning)
// - Duplicate: all neighbours hold the value (needs broadcast at the end)
//
// Note: this assumes plane_dim >= row count and plane_dim % row count == 0
pub struct DummyStageState<E: Float> {
    // Equal m_i'(j-1)
    m: E,
    l: E,
}
