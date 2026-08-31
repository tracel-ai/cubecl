use crate::shared::Architecture;

pub use cubecl_core::ir::amd::{AMDArchitecture, AmdWmma};

impl Architecture for AMDArchitecture {
    fn warp_size(&self) -> u32 {
        // Zero for an architecture the table does not know, which the HIP runtime checks the
        // driver's own answer against before anything is compiled for it.
        self.plane_dim().unwrap_or(0)
    }

    fn is_wmma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 | AMDArchitecture::GFX12
        )
    }

    fn is_mfma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94
        )
    }
}
