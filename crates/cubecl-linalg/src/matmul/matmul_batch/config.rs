use crate::matmul::config::{ComptimeConfig, MatmulConfig, MatmulLaunchConfig};
use crate::matmul::matmul_global::GmmConfig;

pub trait BmmConfig: ComptimeConfig + MatmulConfig + MatmulLaunchConfig {
    type GmmConfig: GmmConfig;

    fn to_gmm_config(&self) -> Self::GmmConfig;
}
