use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_common::{rand::get_seeded_rng, stub::Mutex};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
}

/// Pseudo-random generator
pub(crate) fn random<P: PrngRuntime<E>, R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    prng: P,
    output: &mut TensorHandleRef<'_, R>,
) {
    let seeds = get_seeds();
    let args = prng.args();

    let cube_dim = CubeDim::default();
    let cube_count = prng_cube_count(output.size(), cube_dim, N_VALUES_PER_THREAD);

    let output = output.as_tensor_arg(1);

    prng_kernel::launch::<P, E, R>(
        client,
        cube_count,
        cube_dim,
        output,
        ScalarArg::new(seeds[0]),
        ScalarArg::new(seeds[1]),
        ScalarArg::new(seeds[2]),
        ScalarArg::new(seeds[3]),
        args,
        N_VALUES_PER_THREAD as u32,
    );
}

fn prng_cube_count(num_elems: usize, cube_dim: CubeDim, n_values_per_thread: usize) -> CubeCount {
    let num_threads = f32::ceil(num_elems as f32 / n_values_per_thread as f32);
    let num_invocations = f32::ceil(num_threads / cube_dim.num_elems() as f32);
    let cubes_x = f32::ceil(f32::sqrt(num_invocations));
    let cubes_y = f32::ceil(num_invocations / cubes_x);

    CubeCount::Static(cubes_x as u32, cubes_y as u32, 1)
}

pub(crate) fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock().unwrap();
    let mut rng: StdRng = match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.random());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}

pub(crate) trait PrngArgs<E: CubeElement>: Send + Sync + 'static {
    type Args: LaunchArg;

    fn args<'a, R: Runtime>(self) -> <Self::Args as LaunchArg>::RuntimeArg<'a, R>;
}

#[cube]
pub(crate) trait PrngRuntime<E: CubeElement + CubeType>:
    Send + Sync + 'static + PrngArgs<E>
{
    #[allow(clippy::too_many_arguments)]
    fn inner_loop(
        args: Self::Args,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<E>,
    );
}

#[cube(launch)]
fn prng_kernel<P: PrngRuntime<E>, E: CubeElement + Numeric>(
    output: &mut Tensor<E>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: P::Args,
    #[comptime] n_values_per_thread: u32,
) {
    let cube_offset = CUBE_POS * CUBE_DIM;

    let write_index_base = cube_offset * n_values_per_thread + UNIT_POS;

    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;

    // Creation of n_values_per_thread values, specific to the distribution
    P::inner_loop(
        args,
        write_index_base,
        CUBE_DIM,
        n_values_per_thread,
        &mut state_0,
        &mut state_1,
        &mut state_2,
        &mut state_3,
        output,
    );
}

#[cube]
pub(crate) fn taus_step_0(z: u32) -> u32 {
    taus_step(z, 13u32, 19u32, 12u32, 4294967294u32)
}

#[cube]
pub(crate) fn taus_step_1(z: u32) -> u32 {
    taus_step(z, 2u32, 25u32, 4u32, 4294967288u32)
}

#[cube]
pub(crate) fn taus_step_2(z: u32) -> u32 {
    taus_step(z, 3u32, 11u32, 17u32, 4294967280u32)
}

#[cube]
fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = z << s1;
    let b = b ^ z;
    let b = b >> s2;
    let z = (z & m) << s3;
    z ^ b
}

#[cube]
pub(crate) fn lcg_step(z: u32) -> u32 {
    let a = 1664525u32;
    let b = 1013904223u32;

    z * a + b
}

#[cube]
pub(crate) fn to_probability(int_random: u32) -> f32 {
    let tmp = 2.328_306_4e-10f32;
    f32::cast_from(int_random) * tmp
}
