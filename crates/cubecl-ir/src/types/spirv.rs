use alloc::vec::Vec;

use cubecl_macros_internal::TypeHash;
use pliron::derive::{format, pliron_type};

use crate::aligned;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format]
pub enum ClampMode {
    Undefined,
    #[format("` ` $0")]
    Constant(u32),
    ClampToEdge,
    Repeat,
    RepeatMirrored,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "spirv.tensor_layout",
    format = "`<` $rank `d, clamp: ` $clamp_mode `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct TensorLayoutType {
    pub rank: usize,
    pub clamp_mode: ClampMode,
}
aligned!(TensorLayoutType, align_of::<u64>()); //Dummy align, ignored in SPIR-V

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "spirv.tensor_view",
    format = "`<` $rank `d, has_dims: ` $has_dims `[` vec($permutation, Char(`,`)) `]`",
    generate_get = true,
    verifier = "succ"
)]
pub struct TensorViewType {
    pub rank: usize,
    pub has_dims: bool,
    pub permutation: Vec<usize>,
}
aligned!(TensorViewType, align_of::<u64>()); //Dummy align, ignored in SPIR-V
