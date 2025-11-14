use cubecl_core::CubeDim;

use crate::components::{
    MatmulIdent, MatmulLineSizes, TilingScheme, error::MatmulSetupError,
    global::multi_stage::LoadMaxRoundPlaneCount,
};

pub(crate) fn cube_dim_validation(cube_dim: CubeDim) -> Result<(), MatmulSetupError> {
    #[cfg(target_os = "macos")]
    {
        if cube_dim.num_elems() >= 512 {
            use crate::components::error::MatmulAvailabilityError;

            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CubeDimTooBig(cube_dim),
            ));
        }
    }

    Ok(())
}

/// Maximal number of planes each reader can handle to divide its workload evenly
pub struct MaxGlobalReaderPlanes {
    pub lhs: u32,
    pub rhs: u32,
}

impl MaxGlobalReaderPlanes {
    /// Create a MaxGlobalReaderPlanes
    pub fn new<LL: LoadMaxRoundPlaneCount, RL: LoadMaxRoundPlaneCount>(
        tiling_scheme: &TilingScheme,
        line_sizes: &MatmulLineSizes,
        plane_dim: u32,
    ) -> Self {
        MaxGlobalReaderPlanes {
            lhs: LL::max_round_plane_count(
                tiling_scheme,
                MatmulIdent::Lhs,
                line_sizes.lhs,
                plane_dim,
            ),
            rhs: RL::max_round_plane_count(
                tiling_scheme,
                MatmulIdent::Rhs,
                line_sizes.rhs,
                plane_dim,
            ),
        }
    }
}
