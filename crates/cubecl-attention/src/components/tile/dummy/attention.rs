use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::Tile;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords3d;
use std::marker::PhantomData;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::tile::TileAttention;
use crate::components::tile::dummy::{
    FlashMatmul, FlashMatmulConfig, FlashPrecision, ScoreFragment,
};
use crate::components::{
    AttentionPrecision,
    global::GlobalAttentionConfig,
    tile::dummy::{AccumulatorFragment, DummyWriter, KeyValueFragment, QueryFragment},
};

pub struct DummyTileAttention<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    _phantom: PhantomData<(FP, FM)>,
}

#[cube]
impl<AP: AttentionPrecision, FM: FlashMatmul<AP::FlashPrecision>> TileAttention<AP>
    for DummyTileAttention<AP::FlashPrecision, FM>
{
    type Config = FM::Config;

    type Writer = DummyWriter<AP::EO>;

    type State = RunningState<AP::EA>;

    type Query = QueryFragment<AP::FlashPrecision, FM>;
    type KeyValue = KeyValueFragment<AP::FlashPrecision, FM>;
    type Score = ScoreFragment<AP::FlashPrecision, FM>;
    type Accumulator = AccumulatorFragment<AP::FlashPrecision, FM>;

    fn execute(
        key_tile: &Tile<AP::ES>,
        value_tile: &Tile<AP::ES>,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    ) {
        comment!("Tile: Execute");
        let inv_sqrt_dk = AP::EA::new(comptime!(
            1.0 / (config.attention_tile_size().head_dim as f32).sqrt()
        ));

        let prev_m = state.m;
        let prev_l = state.l;

        FM::fill_key_value(key_tile, key_value.key_mut(), config);

        FM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut score_prob.fragment,
            config,
        );

        score_prob.multiply_score(inv_sqrt_dk);
        let new_m = score_prob.row_max(prev_m);
        score_prob.to_prob(new_m);
        let row_sum = score_prob.row_sum();

        let exp_m_diff = Exp::exp(prev_m - new_m);
        let new_l = exp_m_diff * prev_l + row_sum;

        FM::fill_key_value(value_tile, key_value.value_mut(), config);

        accumulator.scale(exp_m_diff);

        FM::value_matmul(
            &score_prob.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );

        state.m = new_m;
        state.l = new_l;
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] _config: Self::Config) {
        acc.scale(AP::EA::recip(state.l));
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
        let mut out_smem =
            SharedMemory::<AP::EA>::new(stage_config.attention_tile_size().accumulator_size());

        FM::write_results::<AP::EA>(
            &acc.fragment,
            &mut out_smem.to_slice_mut().try_cast_unchecked(),
            stage_config,
        );

        DummyWriter::<AP::EO>::write::<G>(
            writer,
            out_smem.to_slice().try_cast_unchecked(),
            0,
            0,
            global_config,
        )
    }

    fn init_writer(out: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer {
        DummyWriter::new(out, 0, 0, 0)
    }

    fn init_fragments(
        query_reader: QueryRegisterReader<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> (Self::Query, Self::KeyValue, Self::Score, Self::Accumulator) {
        (
            Self::Query::new(query_reader, config),
            Self::KeyValue::new(config),
            Self::Score::new(config),
            Self::Accumulator::new(config),
        )
    }
}

#[derive(CubeType)]
pub struct RunningState<E: Float> {
    m: E,
    l: E,
}
