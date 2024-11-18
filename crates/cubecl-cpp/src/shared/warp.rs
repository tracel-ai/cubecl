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
            WarpInstruction::ReduceMax { input, out } => reduce_comparison(f, input, out, "max"),
            WarpInstruction::ReduceMin { input, out } => reduce_comparison(f, input, out, "min"),
            WarpInstruction::Elect { out } => write!(
                f,
                "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
            ),
            WarpInstruction::All { input, out } => {
                write!(
                    f,
                    "
                        {out} = {input};
                    "
                )?;
                for k in 0..out.item().vectorization {
                    let __all = D::warp_all_indexed(out, k);
                    let out_indexed = out.index(k);
                    write!(
                        f,
                        "
                        {{
                            for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
                                {out_indexed} = {__all};
                            }}
                        }}
                        "
                    )?;
                }
                Ok(())
            }
            WarpInstruction::Any { input, out } => {
                write!(
                    f,
                    "
                        {out} = {input};
                    "
                )?;
                for k in 0..out.item().vectorization {
                    let __any = D::warp_any_indexed(out, k);
                    let out_indexed = out.index(k);
                    write!(
                        f,
                        "
                        {{
                            for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
                                {out_indexed} = {__any};
                            }}
                        }}
                        "
                    )?;
                }
                Ok(())
            }
            WarpInstruction::Broadcast { input, id, out } => {
                for k in 0..out.item().vectorization {
                    let __shfl = D::warp_shuffle_indexed(input, k, id);
                    let out_indexed = out.index(k);
                    write!(
                        f,
                        "
                        {{
                            for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
                                {out_indexed} = {__shfl};
                            }}
                        }}
                        "
                    )?;
                }
                Ok(())
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
    write!(
        f,
        "
        {out} = {input};
        "
    )?;
    for k in 0..out.item().vectorization {
        let __shfl_xor = D::warp_shuffle_xor_indexed(out, k);
        let out_indexed = out.index(k);
        write!(
            f,
            "
            {{
                for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{
                   {out_indexed} {op} {__shfl_xor};
                }}
            }}
            "
        )?;
    }
    Ok(())
}

fn reduce_comparison<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    cmp: &str,
) -> core::fmt::Result {
    write!(
        f,
        "
        {out} = {input};
        "
    )?;
    let instruction = match out.elem() {
        Elem::F16 | Elem::BF16 => format!("__h{cmp}"),
        Elem::F162 | Elem::BF162 => format!("__h{cmp}2"),
        _ => cmp.to_string(),
    };
    for k in 0..out.item().vectorization {
        let __shfl_down = D::warp_shuffle_down_indexed(out, k);
        let out_indexed = out.index(k);
        write!(
            f,
            "
            {{
                for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
                    {out_indexed} = {instruction}({out_indexed}, {__shfl_down});
                }}
            }}
            "
        )?;
    }
    Ok(())
}
