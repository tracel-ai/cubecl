use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_common::{rand::get_seeded_rng, stub::Mutex};
use cubecl_std::tensor::{
    View,
    layout::{
        Coords1d,
        linear::{LinearView, linear_view},
    },
};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
}

/// Pseudo-random generator
pub(crate) fn random<F: RandomFamily, E: Numeric, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    prng: F::Runtime<E>,
    output: TensorHandleRef<'_, R>,
) {
    let seeds = get_seeds();
    let args = prng.args();

    let cube_dim = CubeDim::default();
    let cube_count = prng_cube_count(output.size(), cube_dim, N_VALUES_PER_THREAD);

    let output_line_size = 1;
    // TODO: Higher vectorization can add some correlation locally.
    //
    // let output_line_size = tensor_line_size_parallel(
    //     R::line_size_elem(&E::as_elem_native_unchecked()),
    //     output.shape,
    //     output.strides,
    //     output.strides.len() - 1,
    // );

    let output = linear_view(client, &output, &output_line_size);

    prng_kernel::launch::<F, E, R>(
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
        output_line_size as u32,
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

pub(crate) trait PrngArgs<E: Numeric>: Send + Sync + 'static {
    type Args: LaunchArg;

    fn args<'a, R: Runtime>(self) -> <Self::Args as LaunchArg>::RuntimeArg<'a, R>;
}

pub(crate) trait RandomFamily: Send + Sync + 'static + std::fmt::Debug {
    type Runtime<E: Numeric>: PrngRuntime<E>;
}

#[cube]
pub(crate) trait PrngRuntime<E: Numeric>: Send + Sync + 'static + PrngArgs<E> {
    #[allow(clippy::too_many_arguments)]
    fn inner_loop(
        args: Self::Args,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        #[comptime] line_size: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Line<E>, Coords1d, ReadWrite>,
    );
}

type Args<F, E> = <<F as RandomFamily>::Runtime<E> as PrngArgs<E>>::Args;

#[cube(launch)]
fn prng_kernel<F: RandomFamily, E: Numeric>(
    output: &mut LinearView<Line<E>, ReadWrite>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Args<F, E>,
    #[comptime] n_values_per_thread: u32,
    #[comptime] line_size: u32,
) {
    let cube_offset = CUBE_POS * CUBE_DIM;

    let write_index_base = cube_offset * n_values_per_thread / line_size + UNIT_POS;

    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;

    // Creation of n_values_per_thread values, specific to the distribution
    F::Runtime::inner_loop(
        args,
        write_index_base,
        CUBE_DIM,
        n_values_per_thread,
        line_size,
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

/// Converts a `u32` into a `f32` in the unit interval `[0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_closed_open(int_random: u32) -> f32 {
    // Use upper 24 bits for f32 precision
    // https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    let shifted = int_random >> 8;
    f32::cast_from(shifted) / 16777216.0 // 2^24
}

/// Converts a `u32` into a `f32` in the unit interval `(0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_open(int_random: u32) -> f32 {
    // Use upper 23 bits to leave room for the offset
    let shifted = int_random >> 9;
    (f32::cast_from(shifted) + 1.0) / 8388609.0 // 2^23 + 1
}
