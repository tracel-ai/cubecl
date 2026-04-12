use itertools::Itertools;

use super::{Item, Variable};
use std::fmt::Display;

#[derive(Debug, Clone)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subgroup {
    Elect {
        out: Variable,
    },
    All {
        input: Variable,
        out: Variable,
    },
    Any {
        input: Variable,
        out: Variable,
    },
    Ballot {
        input: Variable,
        out: Variable,
    },
    Broadcast {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sum {
        input: Variable,
        out: Variable,
    },
    ExclusiveSum {
        input: Variable,
        out: Variable,
    },
    InclusiveSum {
        input: Variable,
        out: Variable,
    },
    Prod {
        input: Variable,
        out: Variable,
    },
    ExclusiveProd {
        input: Variable,
        out: Variable,
    },
    InclusiveProd {
        input: Variable,
        out: Variable,
    },
    Min {
        input: Variable,
        out: Variable,
    },
    Max {
        input: Variable,
        out: Variable,
    },
    Shuffle {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    ShuffleXor {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    ShuffleUp {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    ShuffleDown {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
}

impl Display for Subgroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subgroup::Elect { out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupElect();")
            }
            Subgroup::All { input, out } => {
                let out = out.fmt_left();
                match input.item() {
                    Item::Scalar(_) => writeln!(f, "{out} = subgroupAll({input});"),
                    Item::Vector(_, vector_size) => {
                        let elems = (0..vector_size)
                            .map(|i| format!("subgroupAll({})", input.index(i)))
                            .join(", ");
                        writeln!(f, "{out} = vec{vector_size}({elems});")
                    }
                    _ => panic!("Unsupported item for subgroupAll"),
                }
            }
            Subgroup::Any { input, out } => {
                let out = out.fmt_left();
                match input.item() {
                    Item::Scalar(_) => writeln!(f, "{out} = subgroupAny({input});"),
                    Item::Vector(_, vector_size) => {
                        let elems = (0..vector_size)
                            .map(|i| format!("subgroupAny({})", input.index(i)))
                            .join(", ");
                        writeln!(f, "{out} = vec{vector_size}({elems});")
                    }
                    _ => panic!("Unsupported item for subgroupAny"),
                }
            }
            Subgroup::Broadcast { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupBroadcast({lhs}, {rhs});")
            }
            Subgroup::Ballot { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupBallot({input});")
            }
            Subgroup::Sum { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupAdd({input});")
            }
            Subgroup::ExclusiveSum { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupExclusiveAdd({input});")
            }
            Subgroup::InclusiveSum { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupInclusiveAdd({input});")
            }
            Subgroup::Prod { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMul({input});")
            }
            Subgroup::ExclusiveProd { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupExclusiveMul({input});")
            }
            Subgroup::InclusiveProd { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupInclusiveMul({input});")
            }
            Subgroup::Min { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMin({input});")
            }
            Subgroup::Max { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMax({input});")
            }
            Subgroup::Shuffle { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupShuffle({lhs}, {rhs});")
            }
            Subgroup::ShuffleXor { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupShuffleXor({lhs}, {rhs});")
            }
            Subgroup::ShuffleUp { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupShuffleUp({lhs}, {rhs});")
            }
            Subgroup::ShuffleDown { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupShuffleDown({lhs}, {rhs});")
            }
        }
    }
}
