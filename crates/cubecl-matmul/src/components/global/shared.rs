use crate::{components::global::GlobalConfig, kernels::MatmulSetupError};

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
