use cubecl::prelude::*;
use cubecl_core as cubecl;
use std::f32::consts::PI;

use super::{PrngArgs, PrngRuntime, random};

use crate::{lcg_step, taus_step_0, taus_step_1, taus_step_2, to_probability};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Normal<E: Numeric> {
    mean: E,
    std: E,
}

#[cube]
impl<E: CubeElement + Numeric> PrngRuntime<E> for Normal<E> {
    fn inner_loop(
        args: Normal<E>,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        #[comptime] line_size: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<Line<E>>,
    ) {
        let mean = f32::cast_from(args.mean);
        let std = f32::cast_from(args.std);

        let mut output_line_0 = Line::empty(line_size);
        let mut output_line_1 = Line::empty(line_size);

        let num_iterations = n_values_per_thread / line_size / 2;
        #[unroll(num_iterations <= 8)]
        for line_index in 0..num_iterations {
            // vectorization
            #[unroll]
            for i in 0..line_size {
                // First random uniform integer
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let unit_0 = to_probability(int_random);

                // Second random uniform integer
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let unit_1 = to_probability(int_random);

                // Box-Muller transform
                let coeff = Log::log(unit_0) * -2.0;
                let coeff = Sqrt::sqrt(coeff) * std;
                let trigo_arg = 2.0 * PI * unit_1;

                let normal_0 = f32::cos(trigo_arg) * coeff + mean;
                let normal_1 = f32::sin(trigo_arg) * coeff + mean;

                output_line_0[i] = E::cast_from(normal_0);
                output_line_1[i] = E::cast_from(normal_1);
            }

            let iteration_offset = line_index * n_invocations * 2;
            let write_index_0 = write_index_base + iteration_offset;
            let write_index_1 = write_index_0 + n_invocations;

            output[write_index_0] = output_line_0;
            output[write_index_1] = output_line_1;
        }
    }
}

impl<E: CubeElement + Numeric> PrngArgs<E> for Normal<E> {
    type Args = Self;

    fn args<'a, R: Runtime>(self) -> NormalLaunch<'a, E, R> {
        NormalLaunch::new(ScalarArg::new(self.mean), ScalarArg::new(self.std))
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    mean: E,
    std: E,
    out: TensorHandleRef<R>,
) {
    random(client, Normal { mean, std }, out)
}
