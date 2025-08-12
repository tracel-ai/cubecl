use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    MatrixLayout,
    stage::StageToTileReader,
    tile::{Tile, TileMatmul},
};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    global::dummy::QueryRegisterReader,
    stage::{
        StageAttention, StageAttentionConfig,
        dummy::{AttentionStageMemoryConfig, config::DummyStageConfig},
    },
};

pub struct DummyStageAttention<
    AP: AttentionPrecision,
    // TODO: tile matmul should not hardcode to MatmulPrecision, but be generic over Lhs, Rhs, Acc
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
    type Accumulator = VTM::Accumulator;
    type Writer = DummyWriter;

    type State = DummyStageState<AP::EA>;

    type ScoreTileMatmul = STM;

    // Tc times, each call is at an index j
    // Return (m_ij, l_ij) [new]
    fn execute(
        query_reader: &QueryRegisterReader<AP>,
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        acc: &mut Self::Accumulator,
        prev_state: &Self::State,
        #[comptime] config: Self::Config,
    ) -> Self::State {
        comment!("Stage: Execute");

        let prev_m = prev_state.m;
        let prev_l = prev_state.l;
        let row = UNIT_POS_X / 4;
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        /////////////////////////////////////////////
        ///// Put Q directly in fragment for Score Matmul
        // TODO: very bad to load it at this moment
        let query_fragment = query_reader.read_tile::<STM>(config.score_config());

        /////////////////////////////////////////////
        ///// Put K in fragment from reader for Score Matmul
        let key_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
            AttentionStageMemoryConfig<STM::Config>,
        >(key_reader, 0, 0, config.score_stage_memory_config());
        // TODO: This allocation should be reused in each execution
        let mut key_fragment = STM::allocate_rhs(config.score_config());
        STM::fill_rhs(&key_tile, &mut key_fragment, config.score_config());

        /////////////////////////////////////////////
        ///// Init scores
        // TODO: This allocation should be reused in each execution
        let mut scores =
            <STM as TileMatmul<AP::MatmulPrecision>>::allocate_accumulator(config.score_config());
        // TODO: zeroing must be in global matmul header
        STM::zero_accumulator(&mut scores, config.score_config());

        /////////////////////////////////////////////
        ///// Score matmul S=Q·K+0
        STM::execute(
            &query_fragment,
            &key_fragment,
            &mut scores,
            config.score_config(),
        );

        /////////////////////////////////////////////
        ///// Make sure we work with the right registers for scores
        // TODO work on scores register directly
        let mut tmp_smem = SharedMemory::<AP::EA>::new(64);
        STM::write_results(
            &scores,
            &mut tmp_smem.to_slice_mut().try_cast_unchecked(),
            config.score_config(),
        );
        let s0 = tmp_smem[index_0];
        let s1 = tmp_smem[index_1];

        /////////////////////////////////////////////
        ///// 𝑚 ( 𝑗) 𝑖 = max(𝑚 ( 𝑗−1) 𝑖 ,rowmax(S ( 𝑗) 𝑖))
        let mut m = prev_m;
        for i in 0..8 {
            let ts = tmp_smem[row * 8 + i];
            if ts > m {
                m = ts;
            }
        }

        /////////////////////////////////////////////
        ///// ℓ ( 𝑗) 𝑖 = 𝑒 𝑚 𝑗−1 𝑖 −𝑚 ( 𝑗) 𝑖 ℓ ( 𝑗−1) 𝑖 + rowsum(P˜ ( 𝑗) 𝑖)
        let epm = Exp::exp(prev_m - m);
        let mut rowsum = AP::EA::from_int(0);
        for i in 0..8 {
            rowsum += tmp_smem[row * 8 + i];
        }
        let l = epm * prev_l + rowsum;

        /////////////////////////////////////////////
        ///// Compute P and put it to fragment for Value Matmul
        tmp_smem[index_0] = Exp::exp(s0 - m);
        tmp_smem[index_1] = Exp::exp(s1 - m);
        let p = Tile::<AP::ES> {
            slice: tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };
        // TODO: This allocation should be reused in each execution
        let mut p_fragment = VTM::allocate_lhs(config.value_config());
        VTM::fill_lhs(&p, &mut p_fragment, config.value_config());

        /////////////////////////////////////////////
        ///// Put V in fragment from reader for Value Matmul
        let value = <R as StageToTileReader<AP::ES>>::read_tile::<
            AttentionStageMemoryConfig<VTM::Config>,
        >(value_reader, 0, 0, config.value_stage_memory_config());
        // TODO: This allocation should be reused in each execution
        let mut v_fragment = VTM::allocate_rhs(config.value_config());
        VTM::fill_rhs(&value, &mut v_fragment, config.value_config());

        /////////////////////////////////////////////
        ///// Scale acc by epm
        // TODO modify registers directly when we are certain we are in the right row
        // Instead of storing modifying then refilling
        VTM::write_results(
            acc,
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
        VTM::fill_accumulator(&tile, acc, config.value_config());

        /////////////////////////////////////////////
        ///// Value Matmul O = P·V + scaled_O
        VTM::execute(&p_fragment, &v_fragment, acc, config.value_config());

        DummyStageState::<AP::EA> { m, l }
    }

    fn last_update(acc: &mut Self::Accumulator, prev_state: Self::State) {
        comment!("Stage: Last Update");
        // O_i = 1/diag(l_i_Tc) x O_i_Tc
        // todo!()
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::State {
        comment!("Stage: Init Stage");

        DummyStageState::<AP::EA> {
            m: AP::EA::NEG_INFINITY,
            l: AP::EA::from_int(0),
        }
    }

    fn write(acc: &Self::Accumulator, writer: Self::Writer) {
        comment!("Stage: Write");
        // todo
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        comment!("Stage: Zero Accumulator");
        VTM::zero_accumulator(acc, config.value_config());
    }

    fn init_writer(out: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer {
        DummyWriter::new()
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let config = config.value_config();
        let mut acc = VTM::allocate_accumulator(config);
        VTM::zero_accumulator(&mut acc, config);
        acc
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

#[derive(CubeType)]
pub struct DummyWriter {}
#[derive(CubeType)]
pub struct DummyAccumulator {}

#[cube]
impl DummyWriter {
    fn new() -> DummyWriter {
        DummyWriter {}
    }
}

#[cube]
impl DummyAccumulator {
    fn new() -> DummyAccumulator {
        DummyAccumulator {}
    }

    fn zero(&self) {}
}
