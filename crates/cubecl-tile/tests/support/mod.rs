//! What the tests share: device inputs with their host copies, tile arguments built from them,
//! and the comparison and outcome a test ends on.

mod correctness;
mod outcome;
mod stubs;
mod test_tensor;
mod tile_input;

pub use correctness::*;
pub use outcome::*;
pub use test_tensor::*;
pub use tile_input::*;
