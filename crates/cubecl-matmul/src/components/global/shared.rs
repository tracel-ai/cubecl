use std::collections::HashMap;

use crate::{
    components::{AvailableLineSizes, InputIdent, TilingScheme, global::GlobalConfig},
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
pub struct LoaderTasks {
    pub lhs: u32,
    pub rhs: u32,
}

/// Number of tasks per plane before knowing the final vectorization
pub struct LoaderTasksMap {
    pub lhs: HashMap<u8, u32>,
    pub rhs: HashMap<u8, u32>,
}

impl LoaderTasksMap {
    pub fn new(
        tiling_scheme: &TilingScheme,
        available_line_sizes: &AvailableLineSizes,
        plane_dim: u32,
    ) -> LoaderTasksMap {
        LoaderTasksMap {
            lhs: num_tasks_per_line_size(
                tiling_scheme,
                InputIdent::Lhs,
                &available_line_sizes.lhs,
                plane_dim,
            ),
            rhs: num_tasks_per_line_size(
                tiling_scheme,
                InputIdent::Rhs,
                &available_line_sizes.rhs,
                plane_dim,
            ),
        }
    }

    pub fn resolve(self, lhs_line_size: u8, rhs_line_size: u8) -> LoaderTasks {
        LoaderTasks {
            lhs: *self
                .lhs
                .get(&lhs_line_size)
                .expect("Selected line size should be associated with a number of tasks"),
            rhs: *self
                .rhs
                .get(&rhs_line_size)
                .expect("Selected line size should be associated with a number of tasks"),
        }
    }
}

// TODO maybe move elsewhere
// And make generic on loader somehow
pub(crate) fn num_tasks_per_line_size(
    tiling_scheme: &TilingScheme,
    ident: InputIdent,
    line_sizes: &[u8],
    plane_dim: u32,
) -> HashMap<u8, u32> {
    line_sizes
        .iter()
        .map(|line_size| {
            (
                *line_size,
                num_tasks(tiling_scheme, ident, *line_size, plane_dim),
            )
        })
        .collect::<HashMap<_, _>>()
}

// TODO different per loader
pub(crate) fn num_tasks(
    tiling_scheme: &TilingScheme,
    ident: InputIdent,
    line_size: u8,
    plane_dim: u32,
) -> u32 {
    let num_elements = tiling_scheme.elements_in_stage(ident);
    num_elements.div_ceil(plane_dim * line_size as u32)
}
