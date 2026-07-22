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
//     WorkgroupUniformLoad {
//         input: Value,
//         out: Value,
//     },
//     BitwiseOr {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     BitwiseAnd {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     BitwiseXor {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     CountBits {
//         input: Value,
//         out: Value,
//     },
//     ReverseBits {
//         input: Value,
//         out: Value,
//     },
//     ShiftLeft {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     ShiftRight {
//         lhs: Value,
//         rhs: Value,
//         out: Value,
//     },
//     BitwiseNot {
//         input: Value,
//         out: Value,
//     },
//     LeadingZeros {
//         input: Value,
//         out: Value,
//     },
//     TrailingZeros {
//         input: Value,
//         out: Value,
//     },
//     FindFirstSet {
//         input: Value,
//         out: Value,
//     },
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
//     Subgroup(Subgroup),
// }

// impl Display for Instruction {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Instruction::WorkgroupBarrier => f.write_str("workgroupBarrier();\n"),
//             Instruction::StorageBarrier => f.write_str("storageBarrier();\n"),
//             Instruction::WorkgroupUniformLoad { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = workgroupUniformLoad({input});")
//             }
//             Instruction::BitwiseOr { lhs, rhs, out } => {
//                 let lhs = lhs.fmt_cast_to(out.item());
//                 let rhs = rhs.fmt_cast_to(out.item());
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = {lhs} | {rhs};")
//             }
//             Instruction::BitwiseAnd { lhs, rhs, out } => {
//                 let lhs = lhs.fmt_cast_to(out.item());
//                 let rhs = rhs.fmt_cast_to(out.item());
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = {lhs} & {rhs};")
//             }
//             Instruction::BitwiseXor { lhs, rhs, out } => {
//                 let lhs = lhs.fmt_cast_to(out.item());
//                 let rhs = rhs.fmt_cast_to(out.item());
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = {lhs} ^ {rhs};")
//             }
//             Instruction::CountBits { input, out } => {
//                 let out_item = out.item();
//                 let out = out.fmt_left();
//                 match input.elem() == *out_item.elem() {
//                     true => writeln!(f, "{out} = countOneBits({input});"),
//                     false => writeln!(f, "{out} = {out_item}(countOneBits({input}));"),
//                 }
//             }
//             Instruction::ReverseBits { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = reverseBits({input});")
//             }
//             Instruction::ShiftLeft { lhs, rhs, out } => {
//                 let lhs = lhs.fmt_cast_to(out.item());
//                 let rhs = rhs.fmt_cast_to(out.item().with_elem(Elem::U32));
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = {lhs} << {rhs};")
//             }
//             Instruction::ShiftRight { lhs, rhs, out } => {
//                 let lhs = lhs.fmt_cast_to(out.item());
//                 let rhs = rhs.fmt_cast_to(out.item().with_elem(Elem::U32));
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = {lhs} >> {rhs};")
//             }
//             Instruction::BitwiseNot { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = ~{input};")
//             }
//             Instruction::LeadingZeros { input, out } => {
//                 let out_fmt = out.fmt_left();
//                 match input.elem() {
//                     // 64-bit polyfill: split into upper/lower 32 bits
//                     Elem::I64 | Elem::U64 => {
//                         let u64_item = input.item().with_elem(Elem::U64);
//                         let u32_item = input.item().with_elem(Elem::U32);
//                         let input = input.fmt_cast_to(u64_item);
//                         writeln!(
//                             f,
//                             "{out_fmt} = select(countLeadingZeros({u32_item}({input} >> {u32_item}(32u))), 32u + countLeadingZeros({u32_item}({input})), ({input} >> {u32_item}(32u)) == {u64_item}(0));"
//                         )
//                     }
//                     _ => {
//                         let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
//                         writeln!(f, "{out_fmt} = countLeadingZeros({input});")
//                     }
//                 }
//             }
//             Instruction::TrailingZeros { input, out } => {
//                 let out_fmt = out.fmt_left();
//                 match input.elem() {
//                     // 64-bit polyfill: split into upper/lower 32 bits
//                     Elem::I64 | Elem::U64 => {
//                         let u64_item = input.item().with_elem(Elem::U64);
//                         let u32_item = input.item().with_elem(Elem::U32);
//                         let input = input.fmt_cast_to(u64_item);
//                         writeln!(
//                             f,
//                             "{out_fmt} = select(countTrailingZeros({u32_item}({input})), 32u + countTrailingZeros({u32_item}({input} >> {u32_item}(32u))), {u32_item}({input}) == {u32_item}(0u));"
//                         )
//                     }
//                     _ => {
//                         let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
//                         writeln!(f, "{out_fmt} = countTrailingZeros({input});")
//                     }
//                 }
//             }
//             Instruction::FindFirstSet { input, out } => {
//                 let out_fmt = out.fmt_left();
//                 match input.elem() {
//                     // 64-bit polyfill: split into upper/lower 32 bits
//                     Elem::I64 | Elem::U64 => {
//                         let u64_item = input.item().with_elem(Elem::U64);
//                         let u32_item = input.item().with_elem(Elem::U32);
//                         let input = input.fmt_cast_to(u64_item);
//                         writeln!(
//                             f,
//                             "{out_fmt} = select(firstTrailingBit({u32_item}({input})) + 1, select(firstTrailingBit({u32_item}({input} >> {u32_item}(32u))) + 33, {u32_item}(0u), ({input} >> {u32_item}(32u)) == {u64_item}(0)), {u32_item}({input}) == {u32_item}(0u));"
//                         )
//                     }
//                     _ => {
//                         let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
//                         writeln!(f, "{out_fmt} = firstTrailingBit({input}) + 1;")
//                     }
//                 }
//             }
//             Instruction::Subgroup(op) => write!(f, "{op}"),
//             Instruction::Bitcast { input, out } => {
//                 let elem = out.item();
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = bitcast<{elem}>({input});")
//             }
//             Instruction::AtomicLoad { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = atomicLoad({input});")
//             }
//             Instruction::AtomicStore { input, out } => {
//                 writeln!(f, "atomicStore({out},{input});")
//             }
//             Instruction::AtomicSwap { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicExchange({lhs}, {rhs});")
//             }
//             Instruction::AtomicAdd { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicAdd({ptr}, {value});")
//             }
//             Instruction::AtomicSub { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicSub({ptr}, {value});")
//             }
//             Instruction::AtomicMax { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicMax({ptr}, {value});")
//             }
//             Instruction::AtomicMin { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicMin({ptr}, {value});")
//             }
//             Instruction::AtomicAnd { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicAnd({ptr}, {value});")
//             }
//             Instruction::AtomicOr { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicOr({ptr}, {value});")
//             }
//             Instruction::AtomicXor { ptr, value, out } => {
//                 let out = out.fmt_left();
//                 write!(f, "{out} = atomicXor({ptr}, {value});")
//             }
//             Instruction::AtomicCompareExchangeWeak {
//                 ptr,
//                 cmp,
//                 value,
//                 out,
//             } => {
//                 let out = out.fmt_left();
//                 writeln!(
//                     f,
//                     // For compatibility with cuda, only return old_value
//                     "{out} = atomicCompareExchangeWeak({ptr}, {cmp}, {value}).old_value;"
//                 )
//             }
//         }
//     }
// }
