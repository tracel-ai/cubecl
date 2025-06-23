use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl::{CubeLaunch, CubeType, Runtime};

use crate::RandomFamily;

use super::{
    PrngArgs, PrngRuntime, lcg_step, random, taus_step_0, taus_step_1, taus_step_2,
    to_unit_interval_closed_open,
};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Bernoulli<E: Numeric> {
    probability: f32,
    #[cube(comptime)]
    _phantom: PhantomData<E>,
}

#[derive(Debug)]
struct BernoulliFamily;

impl RandomFamily for BernoulliFamily {
    type Runtime<E: Numeric> = Bernoulli<E>;
}

#[cube]
impl<E: Numeric> PrngRuntime<E> for Bernoulli<E> {
    fn inner_loop(
        args: Bernoulli<E>,
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
        let prob = args.probability;

        let mut output_line = Line::empty(line_size);

        let num_iterations = n_values_per_thread / line_size;
        #[unroll(num_iterations <=8)]
        for line_index in 0..num_iterations {
            // vectorization
            #[unroll]
            for i in 0..line_size {
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let float_random = to_unit_interval_closed_open(int_random);
                output_line[i] = E::cast_from(float_random < prob);
            }
            let write_index = line_index * n_invocations + write_index_base;

            output[write_index] = output_line;
        }
    }
}

impl<E: Numeric> PrngArgs<E> for Bernoulli<E> {
    type Args = Self;

    fn args<'a, R: Runtime>(self) -> BernoulliLaunch<'a, E, R> {
        BernoulliLaunch::new(ScalarArg::new(self.probability), &PhantomData::<E>)
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime, E: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    probability: f32,
    out: TensorHandleRef<R>,
) {
    assert_eq!(
        out.elem_size as u32,
        E::elem_size(),
        "Tensor element type must be the same as type E"
    );

    random::<BernoulliFamily, E, R>(
        client,
        Bernoulli::<E> {
            probability,
            _phantom: PhantomData,
        },
        out,
    )
}
