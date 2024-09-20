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
            Subgroup::Elect { out } => write!(f, "{out} = subgroupElect();\n"),
            Subgroup::All { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupAll({input});\n")
            }
            Subgroup::Any { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupAny({input});\n")
            }
            Subgroup::Broadcast { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupBroadcast({lhs}, {rhs});\n")
            }
            Subgroup::Sum { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupAdd({input});\n")
            }
            Subgroup::Prod { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupMul({input});\n")
            }
            Subgroup::Min { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupMin({input});\n")
            }
            Subgroup::Max { input, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = subgroupMax({input});\n")
            }
        }
    }
}
