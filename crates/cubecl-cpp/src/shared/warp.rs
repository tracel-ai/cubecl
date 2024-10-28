use std::fmt::Display;

use crate::shared::{Component, Elem};

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
            WarpInstruction::ReduceMax { input, out } => {
                let max = match out.elem() {
                    Elem::F16 | Elem::BF16 => "__hmax",
                    Elem::F162 | Elem::BF162 => "__hmax2",
                    _ => "max",
                };
                let __shfl_down = D::warp_shuffle_down(out);
                write!(
                    f,
                    "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = {max}({out}, {__shfl_down});
}}
}}
                    "
                )
            }
            WarpInstruction::ReduceMin { input, out } => {
                let min = match out.elem() {
                    Elem::F16 | Elem::BF16 => "__hmin",
                    Elem::F162 | Elem::BF162 => "__hmin2",
                    _ => "min",
                };
                let __shfl_down = D::warp_shuffle_down(out);
                write!(
                    f,
                    "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = {min}({out}, {__shfl_down});
}}
}}
                    "
                )
            }
            WarpInstruction::Elect { out } => write!(
                f,
                "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
                )
            }
            WarpInstruction::All { input, out } => {
                let __all = D::warp_all(out);
                write!(
                    f,
                    "
    {out} = {input};
{{
    {out} =  {__all};
}}
"
                )
            }
            WarpInstruction::Any { input, out } => {
                let __any = D::warp_any(out);
                write!(
                    f,
                    "
    {out} = {input};
{{
    {out} =  {__any};
}}
"
                )
            }
            WarpInstruction::Broadcast { input, id, out } => {
                let __shfl = D::warp_shuffle(input, id);
                write!(
                    f,
                    "
{out} = {__shfl};
            "
                )
            }
        }
    }
}

fn reduce_operator<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    op: &str,
) -> core::fmt::Result {
    let __shfl_xor = D::warp_shuffle_xor(out);
    write!(
        f,
        "
    {out} = {input};
{{
    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
       {out} {op} {__shfl_xor};
    }}
}}
"
    )
}
