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
                    Item::Vec2(_) => {
                        writeln!(f, "{out} = vec2(")?;
                        writeln!(f, "    subgroupAll({input}[0]),")?;
                        writeln!(f, "    subgroupAll({input}[1]),")?;
                        writeln!(f, ");")
                    }
                    Item::Vec3(_) => {
                        writeln!(f, "{out} = vec3(")?;
                        writeln!(f, "    subgroupAll({input}[0]),")?;
                        writeln!(f, "    subgroupAll({input}[1]),")?;
                        writeln!(f, "    subgroupAll({input}[2]),")?;
                        writeln!(f, ");")
                    }
                    Item::Vec4(_) => {
                        writeln!(f, "{out} = vec4(")?;
                        writeln!(f, "    subgroupAll({input}[0]),")?;
                        writeln!(f, "    subgroupAll({input}[1]),")?;
                        writeln!(f, "    subgroupAll({input}[2]),")?;
                        writeln!(f, "    subgroupAll({input}[3]),")?;
                        writeln!(f, ");")
                    }
                }
            }
            Subgroup::Any { input, out } => {
                let out = out.fmt_left();
                match input.item() {
                    Item::Scalar(_) => writeln!(f, "{out} = subgroupAny({input});"),
                    Item::Vec2(_) => {
                        writeln!(f, "{out} = vec2(")?;
                        writeln!(f, "    subgroupAny({input}[0]),")?;
                        writeln!(f, "    subgroupAny({input}[1]),")?;
                        writeln!(f, ");")
                    }
                    Item::Vec3(_) => {
                        writeln!(f, "{out} = vec3(")?;
                        writeln!(f, "    subgroupAny({input}[0]),")?;
                        writeln!(f, "    subgroupAny({input}[1]),")?;
                        writeln!(f, "    subgroupAny({input}[2]),")?;
                        writeln!(f, ");")
                    }
                    Item::Vec4(_) => {
                        writeln!(f, "{out} = vec4(")?;
                        writeln!(f, "    subgroupAny({input}[0]),")?;
                        writeln!(f, "    subgroupAny({input}[1]),")?;
                        writeln!(f, "    subgroupAny({input}[2]),")?;
                        writeln!(f, "    subgroupAny({input}[3]),")?;
                        writeln!(f, ");")
                    }
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
        }
    }
}
