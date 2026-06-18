use pliron::derive::pliron_type;

use crate::interfaces::{aligned, sized};

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cuda.tensor_map",
    format = "`tensor_map`",
    generate_get = true,
    verifier = "succ"
)]
pub struct TensorMapType;
aligned!(TensorMapType, 128);
sized!(TensorMapType, 128);
