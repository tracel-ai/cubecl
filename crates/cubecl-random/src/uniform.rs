use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::{lcg_step, taus_step_0, taus_step_1, taus_step_2, to_probability};

use super::{PrngArgs, PrngRuntime, random};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Uniform<E: Numeric> {
    lower_bound: E,
    upper_bound: E,
}

#[cube]
impl<E: CubeElement + Numeric> PrngRuntime<E> for Uniform<E> {
    fn inner_loop(
        args: Uniform<E>,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<E>,
    ) {
        let lower_bound = args.lower_bound;
        let upper_bound = args.upper_bound;

        let should_unroll = n_values_per_thread <= 8;
        let scale = upper_bound - lower_bound;

        #[unroll(should_unroll)]
        for i in 0..n_values_per_thread {
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let f32_random = to_probability(int_random);

            let f32_uniform = f32_random * f32::cast_from(scale) + f32::cast_from(lower_bound);

            let uniform = E::cast_from(f32_uniform);

            let write_index = i * n_invocations + write_index_base;

            output[write_index] = uniform;
        }
    }
}

impl<E: CubeElement + Numeric> PrngArgs<E> for Uniform<E> {
    type Args = Self;

    fn args<'a, R: Runtime>(self) -> UniformLaunch<'a, E, R> {
        UniformLaunch::new(
            ScalarArg::new(self.lower_bound),
            ScalarArg::new(self.upper_bound),
        )
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lower_bound: E,
    upper_bound: E,
    out: &mut TensorHandleRef<R>,
) {
    random(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
    )
}
