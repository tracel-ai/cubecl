use std::fmt::Debug;

#[derive(Debug)]
/// Line size used for each tensor in global memory accesses.
/// Represents the numbe of elements processed per SIMD load/store.
pub struct AttentionLineSizes {
    pub query: u8,
    pub key: u8,
    pub value: u8,
    pub out: u8,
}

#[derive(Clone, Debug)]
/// Candidate line sizes supported for each tensor.
///
/// These lists begin with compiler-supported sizes and are progressively
/// filtered based on problem shape divisibility and hardware constraints.
pub struct AvailableLineSizes {
    pub query: Vec<u8>,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub out: Vec<u8>,
}
