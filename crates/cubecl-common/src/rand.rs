pub use rand::{Rng, SeedableRng, rngs::StdRng};

use rand::distr::StandardUniform;
use rand::prelude::Distribution;

/// Returns a seeded random number generator using entropy.
#[cfg(feature = "std")]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    StdRng::from_os_rng()
}

/// Returns a seeded random number generator using a pre-generated seed.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    const CONST_SEED: u64 = 42;
    StdRng::seed_from_u64(CONST_SEED)
}

/// Generates random data from a thread-local RNG.
#[cfg(feature = "std")]
#[inline]
pub fn gen_random<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    rand::rng().random()
}

/// Generates random data from a mutex-protected RNG.
#[cfg(not(feature = "std"))]
#[inline]
pub fn gen_random<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    use crate::stub::Mutex;
    static RNG: Mutex<Option<StdRng>> = Mutex::new(None);
    let mut rng = RNG.lock().unwrap();
    if rng.is_none() {
        *rng = Some(get_seeded_rng());
    }
    rng.as_mut().unwrap().random()
}
