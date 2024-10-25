use std::fmt::Display;

use super::{Dialect, Variable};

#[derive(Clone, Debug)]
pub enum WarpInstruction<D: Dialect> {
    ReduceSum {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceProd {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceMax {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceMin {
        input: Variable<D>,
        out: Variable<D>,
    },
    Elect {
        out: Variable<D>,
    },
    All {
        input: Variable<D>,
        out: Variable<D>,
    },
    Any {
        input: Variable<D>,
        out: Variable<D>,
    },
    Broadcast {
        input: Variable<D>,
        id: Variable<D>,
        out: Variable<D>,
    },
}

impl<D: Dialect> Display for WarpInstruction<D> {
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

fn reduce_operator<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
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
