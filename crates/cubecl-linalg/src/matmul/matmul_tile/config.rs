use crate::matmul::{
    config::MatmulConfig,
    matrix::{Ident, MatrixLayout},
};

pub trait TmmConfig: MatmulConfig {
    fn plane_dim(&self) -> u32;
    fn layout(&self, ident: Ident) -> MatrixLayout;
}
