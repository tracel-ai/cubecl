pub mod bernoulli;
pub mod normal;
pub mod uniform;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_random {
    () => {
        use cubecl::prelude::*;
        use cubecl_core as cubecl;

        use cubecl::{client::ComputeClient, prelude::TensorHandleRef};
        use cubecl_linalg::tensor::TensorHandle;
        use cubecl_random::*;

        use core::f32;

        cubecl_random::testgen_random_bernoulli!();
        cubecl_random::testgen_random_normal!();
        cubecl_random::testgen_random_uniform!();
    };
}
