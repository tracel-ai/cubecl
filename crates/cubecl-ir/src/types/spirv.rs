use alloc::vec::Vec;

use cubecl_macros_internal::TypeHash;
use pliron::derive::{format, pliron_type};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format]
pub enum ClampMode {
    #[format("`undefined`")]
    Undefined,
    #[format("`constant(` $0 `)`")]
    Constant(u32),
    #[format("`clamp_to_edge`")]
    ClampToEdge,
    #[format("`repeat`")]
    Repeat,
    #[format("`repeat_mirrored`")]
    RepeatMirrored,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "spirv.tensor_layout",
    format = "`tensor_layout<` $rank `d, clamp: ` $clamp_mode `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct TensorLayoutType {
    pub rank: usize,
    pub clamp_mode: ClampMode,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "spirv.tensor_view",
    format = "`tensor_view<` $rank `d, has_dims: ` $has_dims `[` vec($permutation, Char(`,`)) `]`",
    generate_get = true,
    verifier = "succ"
)]
pub struct TensorViewType {
    pub rank: usize,
    pub has_dims: bool,
    pub permutation: Vec<usize>,
}
