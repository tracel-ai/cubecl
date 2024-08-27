mod array;
mod slice;
mod tensor;

pub use array::*;
pub use slice::*;
pub use tensor::*;

use super::SquareType;

pub trait Container {
    type Item: SquareType;
}
