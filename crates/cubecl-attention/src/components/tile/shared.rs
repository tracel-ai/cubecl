use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::TileConfig;

use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub struct ScoreProb<E: Numeric> {
    fragment: cmma::Matrix<E>,
}

#[cube]
impl<E: Numeric> ScoreProb<E> {
    pub fn init_as_score<S: StageAttentionConfig>(
        #[comptime] score_config: S::ScoreConfig,
        #[comptime] _value_config: S::ValueConfig,
    ) -> ScoreProb<E> {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = (*score_config.tile_size()).into(); (m, n, k)};
        ScoreProb::<E> {
            fragment: unsafe {
                cmma::Matrix::<E>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    m,
                    n,
                    k,
                    cmma::MatrixLayout::RowMajor,
                )
            },
        }
    }

    pub fn as_score(&self) -> cmma::Matrix<E> {
        self.fragment
    }
}

#[derive(CubeType)]
pub struct KeyValue<E: Numeric> {
    fragment: cmma::Matrix<E>,
}

#[cube]
impl<E: Numeric> KeyValue<E> {
    pub fn init_as_key<S: StageAttentionConfig>(
        #[comptime] score_config: S::ScoreConfig,
        #[comptime] _value_config: S::ValueConfig,
    ) -> ScoreProb<E> {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = (*score_config.tile_size()).into(); (m, n, k)};
        ScoreProb::<E> {
            fragment: unsafe {
                cmma::Matrix::<E>::uninitialized(
                    cmma::MatrixIdent::B,
                    m,
                    n,
                    k,
                    cmma::MatrixLayout::RowMajor,
                )
            },
        }
    }

    pub fn as_key(&self) -> cmma::Matrix<E> {
        self.fragment
    }
}
