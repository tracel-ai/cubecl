use crate::matmul::matmul_modular::{
    config::MatmulConfig,
    matrix::{Ident, MatrixLayout},
};

/// Configuration for the Tile matmul (TMM) level
pub trait TmmConfig: MatmulConfig {
    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the line size for the given ident
    fn line_size(&self, ident: Ident) -> u32;
}
