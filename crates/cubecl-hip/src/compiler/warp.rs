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
    Broadcast {
        input: Variable,
        id: Variable,
        out: Variable,
    },
}

impl Display for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarpInstruction::ReduceSum { input, out } => reduce_operator(f, input, out, "+="),
            WarpInstruction::ReduceProd { input, out } => reduce_operator(f, input, out, "*="),
            WarpInstruction::ReduceMax { input, out } => write!(
                f,
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = max({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            ),
            WarpInstruction::ReduceMin { input, out } => write!(
                f,
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = min({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            ),
            WarpInstruction::Elect { out } => write!(
                f,
                "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
            ),
            WarpInstruction::All { input, out } => write!(
                f,
                "
    {out} = {input};
{{
    {out} =  __all_sync(0xFFFFFFFF, {out});
}}
"
            ),
            WarpInstruction::Any { input, out } => write!(
                f,
                "
    {out} = {input};
{{
    {out} =  __any_sync(0xFFFFFFFF, {out});
}}
"
            ),
            WarpInstruction::Broadcast { input, id, out } => write!(
                f,
                "
{out} = __shfl_sync(0xFFFFFFFF, {input}, {id});
            "
            ),
        }
    }
}

fn reduce_operator(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable,
    out: &Variable,
    op: &str,
) -> core::fmt::Result {
    write!(
        f,
        "
    {out} = {input};
{{
    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
       {out} {op} __shfl_xor_sync(-1, {out}, offset);
    }}
}}
"
    )
}
