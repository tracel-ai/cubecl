mod array;
mod sequence;
mod slice;
mod tensor;

pub use array::*;
pub use sequence::*;
pub use slice::*;
pub use tensor::*;

use super::SquareType;

pub trait Container {
    type Item: SquareType;
}
