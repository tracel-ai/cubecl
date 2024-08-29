use std::fmt::Display;

use super::Variable;

#[derive(Clone, Debug)]
pub enum WarpInstruction {
    ReduceSum {
        input: Variable,
        out: Variable,
    },
    ReduceProd {
        input: Variable,
        out: Variable,
    },
    ReduceMax {
        input: Variable,
        out: Variable,
    },
    ReduceMin {
        input: Variable,
        out: Variable,
    },
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
    And {
        input: Variable,
        out: Variable,
    },
    Or {
        input: Variable,
        out: Variable,
    },
    Xor {
        input: Variable,
        out: Variable,
    },
    Broadcast {
        input: Variable,
        id: Variable,
        out: Variable,
    },
}

impl Display for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarpInstruction::ReduceSum { input, out } => reduce_operator(f, input, out, "+"),
            WarpInstruction::ReduceProd { input, out } => reduce_operator(f, input, out, "*"),
            WarpInstruction::ReduceMax { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = max({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            )),
            WarpInstruction::ReduceMin { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = min({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            )),
            WarpInstruction::Elect { out } => f.write_fmt(format_args!(
                "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
            )),
            WarpInstruction::All { input, out } => reduce_operator(f, input, out, "&&"),
            WarpInstruction::Any { input, out } => reduce_operator(f, input, out, "||"),
            WarpInstruction::And { input, out } => reduce_operator(f, input, out, "&"),
            WarpInstruction::Or { input, out } => reduce_operator(f, input, out, "|"),
            WarpInstruction::Xor { input, out } => reduce_operator(f, input, out, "^"),
            WarpInstruction::Broadcast { input, id, out } => f.write_fmt(format_args!(
                "
{out} = __shfl_sync(0xFFFFFFFF, {input}, {id});
            "
            )),
        }
    }
}

fn reduce_operator(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable,
    out: &Variable,
    op: &str,
) -> core::fmt::Result {
    f.write_fmt(format_args!(
        "
    {out} = {input};
                    {{
    for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
        {out} = {out} {op} __shfl_down_sync(0xFFFFFFFF, {out}, offset);
    }}
    }}
                        "
    ))
}
