use super::Variable;
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
    Broadcast {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sum {
        input: Variable,
        out: Variable,
    },
    Prod {
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
}

impl Display for Subgroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Elect { out } => writeln!(f, "{out} = subgroupElect();"),
            Self::All { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupAll({input});")
            }
            Self::Any { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupAny({input});")
            }
            Self::Broadcast { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupBroadcast({lhs}, {rhs});")
            }
            Self::Sum { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupAdd({input});")
            }
            Self::Prod { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMul({input});")
            }
            Self::Min { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMin({input});")
            }
            Self::Max { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = subgroupMax({input});")
            }
        }
    }
}
