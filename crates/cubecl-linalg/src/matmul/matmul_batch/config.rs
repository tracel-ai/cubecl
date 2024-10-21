use crate::matmul::config::{ComptimeConfig, MatmulConfig, MatmulLaunchConfig};
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::stage_dim::StageDim;

pub trait BmmConfig: ComptimeConfig + MatmulConfig + MatmulLaunchConfig {
    type GmmConfig: GmmConfig;

    fn to_gmm_config(&self) -> Self::GmmConfig;
    fn stage_dim(&self, ident: Ident) -> StageDim;
}
