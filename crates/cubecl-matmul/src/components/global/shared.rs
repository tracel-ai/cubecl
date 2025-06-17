use crate::{
    components::{
        InputIdent, MatmulLineSizes, TilingScheme,
        global::{GlobalConfig, multi_stage::LoadMaxRoundPlaneCount},
    },
    kernels::MatmulSetupError,
};

pub(crate) fn shared_global_config_validation<G: GlobalConfig>(
    config: G,
) -> Result<G, MatmulSetupError> {
    #[cfg(target_os = "macos")]
    {
        let cube_dim = config.cube_dim();
        if cube_dim.num_elems() >= 512 {
            use crate::kernels::MatmulAvailabilityError;

            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CubeDimTooBig(cube_dim),
            ));
        }
    }

    Ok(config)
}

/// Number of tasks per plane.
///
/// Example: loading 1024 elements with a line size of 8 gives 128 lines (1024 / 8).
/// With `plane_dim` set to 32, there are 4 tasks (128 / 32).
pub struct MaxLoaders {
    pub lhs: u32,
    pub rhs: u32,
}

impl MaxLoaders {
    pub fn new<LL: LoadMaxRoundPlaneCount, RL: LoadMaxRoundPlaneCount>(
        tiling_scheme: &TilingScheme,
        line_sizes: &MatmulLineSizes,
        plane_dim: u32,
    ) -> Self {
        MaxLoaders {
            lhs: LL::max_round_plane_count(
                tiling_scheme,
                InputIdent::Lhs,
                line_sizes.lhs,
                plane_dim,
            ),
            rhs: RL::max_round_plane_count(
                tiling_scheme,
                InputIdent::Rhs,
                line_sizes.rhs,
                plane_dim,
            ),
        }
    }
}
