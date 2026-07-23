// use itertools::Itertools;

// use super::{Item, Value};
// use std::fmt::Display;

// #[derive(Debug, Clone)]
// #[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
// pub enum Subgroup {
//     Elect { out: Value },
//     All { input: Value, out: Value },
//     Any { input: Value, out: Value },
//     Ballot { input: Value, out: Value },
//     Broadcast { lhs: Value, rhs: Value, out: Value },
//     Sum { input: Value, out: Value },
//     ExclusiveSum { input: Value, out: Value },
//     InclusiveSum { input: Value, out: Value },
//     Prod { input: Value, out: Value },
//     ExclusiveProd { input: Value, out: Value },
//     InclusiveProd { input: Value, out: Value },
//     Min { input: Value, out: Value },
//     Max { input: Value, out: Value },
//     Shuffle { lhs: Value, rhs: Value, out: Value },
//     ShuffleXor { lhs: Value, rhs: Value, out: Value },
//     ShuffleUp { lhs: Value, rhs: Value, out: Value },
//     ShuffleDown { lhs: Value, rhs: Value, out: Value },
// }

// impl Display for Subgroup {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Subgroup::Elect { out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupElect();")
//             }
//             Subgroup::All { input, out } => {
//                 let out = out.fmt_left();
//                 match input.item() {
//                     Item::Scalar(_) => writeln!(f, "{out} = subgroupAll({input});"),
//                     Item::Vector(_, vector_size) => {
//                         let elems = (0..vector_size)
//                             .map(|i| format!("subgroupAll({})", input.index(i)))
//                             .join(", ");
//                         writeln!(f, "{out} = vec{vector_size}({elems});")
//                     }
//                     _ => panic!("Unsupported item for subgroupAll"),
//                 }
//             }
//             Subgroup::Any { input, out } => {
//                 let out = out.fmt_left();
//                 match input.item() {
//                     Item::Scalar(_) => writeln!(f, "{out} = subgroupAny({input});"),
//                     Item::Vector(_, vector_size) => {
//                         let elems = (0..vector_size)
//                             .map(|i| format!("subgroupAny({})", input.index(i)))
//                             .join(", ");
//                         writeln!(f, "{out} = vec{vector_size}({elems});")
//                     }
//                     _ => panic!("Unsupported item for subgroupAny"),
//                 }
//             }
//             Subgroup::Broadcast { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupBroadcast({lhs}, {rhs});")
//             }
//             Subgroup::Ballot { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupBallot({input});")
//             }
//             Subgroup::Sum { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupAdd({input});")
//             }
//             Subgroup::ExclusiveSum { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupExclusiveAdd({input});")
//             }
//             Subgroup::InclusiveSum { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupInclusiveAdd({input});")
//             }
//             Subgroup::Prod { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupMul({input});")
//             }
//             Subgroup::ExclusiveProd { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupExclusiveMul({input});")
//             }
//             Subgroup::InclusiveProd { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupInclusiveMul({input});")
//             }
//             Subgroup::Min { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupMin({input});")
//             }
//             Subgroup::Max { input, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupMax({input});")
//             }
//             Subgroup::Shuffle { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupShuffle({lhs}, {rhs});")
//             }
//             Subgroup::ShuffleXor { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupShuffleXor({lhs}, {rhs});")
//             }
//             Subgroup::ShuffleUp { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupShuffleUp({lhs}, {rhs});")
//             }
//             Subgroup::ShuffleDown { lhs, rhs, out } => {
//                 let out = out.fmt_left();
//                 writeln!(f, "{out} = subgroupShuffleDown({lhs}, {rhs});")
//             }
//         }
//     }
// }
