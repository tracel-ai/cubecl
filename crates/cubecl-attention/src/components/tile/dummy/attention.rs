use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{MatrixLayout, tile::Tile};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::tile::{ScoreMatmul, TileAttention, TileAttentionConfig, ValueMatmul};
use crate::components::{
    AttentionPrecision,
    global::GlobalAttentionConfig,
    tile::dummy::{
        AccumulatorFragment, DummyWriter, KeyValueFragment, QueryFragment, ScoreProbFragment,
    },
};

pub struct DummyTileAttention<
    AP: AttentionPrecision,
    SM: ScoreMatmul<AP>,
    VM: ValueMatmul<AP>,
    T: TileAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>,
> {
    _phantom: PhantomData<(AP, SM, VM, T)>,
}

#[cube]
impl<
    SM: ScoreMatmul<AP>,
    VM: ValueMatmul<AP>,
    AP: AttentionPrecision,
    S: TileAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>,
> TileAttention<AP> for DummyTileAttention<AP, SM, VM, S>
{
    type Config = S;

    type Writer = DummyWriter<AP::EO>;

    type State = RunningState<AP::EA>;

    type Query = QueryFragment<AP, SM>;
    type KeyValue = KeyValueFragment<AP, SM, VM>;
    type ScoreProb = ScoreProbFragment<AP, SM, VM>;
    type Accumulator = AccumulatorFragment<AP, VM>;

    fn execute(
        key_tile: &Tile<AP::ES>,
        value_tile: &Tile<AP::ES>,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    ) {
        comment!("Tile: Execute");
        // 1/sqrt(8)
        let inv_sqrt_dk = AP::EA::new(0.35355);

        let prev_m = state.m;
        let prev_l = state.l;

        SM::fill_rhs(key_tile, key_value.key_mut(), config.score_config());

        SM::execute(
            &query.fragment,
            key_value.key(),
            score_prob.score_mut(),
            config.score_config(),
        );

        score_prob.multiply_score(inv_sqrt_dk);
        let new_m = score_prob.row_max(prev_m);
        score_prob.to_prob(new_m);
        let row_sum = score_prob.row_sum();

        let exp_m_diff = Exp::exp(prev_m - new_m);
        let new_l = exp_m_diff * prev_l + row_sum;

        VM::fill_rhs(value_tile, key_value.value_mut(), config.value_config());

        accumulator.scale(exp_m_diff);

        VM::execute(
            score_prob.prob(),
            key_value.value(),
            &mut accumulator.fragment,
            config.value_config(),
        );

        state.m = new_m;
        state.l = new_l;
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] config: Self::Config) {
        comment!("Tile: Rescale");
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
        comment!("Tile: Init Stage");

        RunningState::<AP::EA> {
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
        comment!("Tile: Write");
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
pub struct RunningState<E: Float> {
    m: E,
    l: E,
}
