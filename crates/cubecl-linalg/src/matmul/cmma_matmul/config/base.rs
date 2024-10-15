use cubecl_core::prelude::*;

use crate::matmul::subroutine::Config;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CmmaConfig {
    pub out_smem_line_size: u32,
}

impl Config for CmmaConfig {}

impl Init for CmmaConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl CubeType for CmmaConfig {
    type ExpandType = Self;
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self {
            out_smem_line_size: 4u32,
        }
    }
}
