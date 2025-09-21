use std::{fmt::Debug, hash::Hash};

use crate::components::{MatrixLayout, StageIdent, TilingScheme};

pub trait StageMemoryConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Number of planes participating in the main computation flow
    fn num_main_flow_planes(&self) -> u32;

    /// Returns the [TilingScheme]
    fn tiling_scheme(&self) -> TilingScheme;

    /// Returns the line size for the given ident
    fn stage_line_size(&self, ident: StageIdent) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout;

    /// Returns the number of stages for the given input
    fn num_stages(&self, ident: StageIdent) -> u32;
}
