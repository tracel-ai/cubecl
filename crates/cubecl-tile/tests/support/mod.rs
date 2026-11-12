//! What the tests share: device inputs with their host copies, tile tensors built from them,
//! and the comparison and outcome a test ends on.

mod correctness;
mod hand_off;
mod outcome;
mod stubs;
mod test_tensor;
mod tile_input;

pub use correctness::*;
pub use hand_off::*;
pub use outcome::*;
pub use test_tensor::*;
pub use tile_input::*;
