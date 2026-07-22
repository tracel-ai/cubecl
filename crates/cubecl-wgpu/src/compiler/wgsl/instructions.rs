// use cubecl_ir::Id;

// use crate::compiler::wgsl::Builtin;

// use super::{
//     Elem, Subgroup,
//     base::{Item, Value},
// };
// use std::fmt::Display;

// /// All instructions that can be used in a WGSL compute shader.
// #[derive(Debug, Clone)]
// #[allow(dead_code)] // Some variants might not be used with different flags
// pub enum Instruction {
//     WorkgroupBarrier,
//     StorageBarrier,
//     AtomicLoad {
//         input: Value,
//         out: Value,
//     },
//     AtomicStore {
//         input: Value,
//         out: Value,
//     },
//     AtomicSwap {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     AtomicCompareExchangeWeak {
//         ptr: Value,
//         cmp: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicAdd {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicSub {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicMax {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicMin {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicAnd {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicOr {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
//     AtomicXor {
//         ptr: Value,
//         value: Value,
//         out: Value,
//     },
// }

// impl Display for Instruction {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Instruction::WorkgroupBarrier => f.write_str("workgroupBarrier();\n"),
//             Instruction::StorageBarrier => f.write_str("storageBarrier();\n"),
//         }
//     }
// }
