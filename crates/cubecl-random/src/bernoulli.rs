use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl::{CubeElement, CubeLaunch, CubeType, Runtime};

use super::{
    PrngArgs, PrngRuntime, lcg_step, random, taus_step_0, taus_step_1, taus_step_2, to_probability,
};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Bernoulli<E: Numeric> {
    probability: f32,
    #[cube(comptime)]
    _phantom: PhantomData<E>,
}

#[cube]
impl<E: CubeElement + Numeric> PrngRuntime<E> for Bernoulli<E> {
    fn inner_loop(
        args: Bernoulli<E>,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<E>,
    ) {
        let prob = args.probability;

        #[unroll(n_values_per_thread <=8)]
        for i in 0..n_values_per_thread {
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let float_random = to_probability(int_random);
            let write_index = i * n_invocations + write_index_base;

            output[write_index] = E::cast_from(float_random < prob);
        }
    }
}

impl<E: CubeElement + Numeric> PrngArgs<E> for Bernoulli<E> {
    type Args = Self;

    fn args<'a, R: Runtime>(self) -> BernoulliLaunch<'a, E, R> {
        BernoulliLaunch::new(ScalarArg::new(self.probability), &PhantomData::<E>)
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    probability: f32,
    out: &mut TensorHandleRef<R>,
) {
    random(
        client,
        Bernoulli::<E> {
            probability,
            _phantom: PhantomData,
        },
        out,
    )
}
