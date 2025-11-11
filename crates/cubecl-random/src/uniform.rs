use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::View;

use crate::{
    RandomFamily, lcg_step, taus_step_0, taus_step_1, taus_step_2, to_unit_interval_closed_open,
};

use super::{PrngArgs, PrngRuntime, random};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Uniform {
    lower_bound: f32,
    upper_bound: f32,
}

#[derive(Debug)]
struct UniformFamily;

impl RandomFamily for UniformFamily {
    type Runtime = Uniform;
}

#[cube]
impl PrngRuntime for Uniform {
    fn inner_loop<E: Numeric>(
        args: Uniform,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        #[comptime] line_size: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Line<E>, u32, ReadWrite>,
    ) {
        let lower_bound = args.lower_bound;
        let upper_bound = args.upper_bound;

        let scale = upper_bound - lower_bound;

        let mut output_line = Line::empty(line_size);

        let num_iterations = n_values_per_thread / line_size;
        #[unroll(num_iterations <= 8)]
        for line_index in 0..num_iterations {
            // vectorization
            #[unroll]
            for i in 0..line_size {
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let f32_random = to_unit_interval_closed_open(int_random);

                let f32_uniform = f32_random * f32::cast_from(scale) + f32::cast_from(lower_bound);

                let uniform = E::cast_from(f32_uniform);

                output_line[i] = uniform;
            }

            let write_index = line_index * n_invocations + write_index_base;

            output[write_index] = output_line;
        }
    }
}

impl PrngArgs for Uniform {
    type Args = Self;

    fn args<'a, R: Runtime>(self) -> UniformLaunch<'a, R> {
        UniformLaunch::new(
            ScalarArg::new(self.lower_bound),
            ScalarArg::new(self.upper_bound),
        )
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: Runtime>(
    client: &ComputeClient<R::Server>,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorHandleRef<R>,
    dtype: StorageType,
) {
    assert_eq!(
        out.elem_size,
        dtype.size(),
        "Tensor element type must be the same as type E"
    );

    random::<UniformFamily, R>(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
        dtype,
    )
}
