use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::super::{
    base::{Dimensions, Offsets},
    config::CmmaComptimeInfo,
};

#[cube]
pub(crate) trait OutputWriter: Send + Sync + 'static {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        offsets: Offsets,
        dims: Dimensions,
        config: Comptime<CmmaComptimeInfo>,
    );
}
