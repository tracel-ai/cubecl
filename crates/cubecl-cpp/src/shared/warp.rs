use std::fmt::Display;

use crate::shared::{Component, Elem, FmtLeft};

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
            WarpInstruction::All { input, out } => reduce_quantifier(f, input, out, D::warp_all),
            WarpInstruction::Any { input, out } => reduce_quantifier(f, input, out, D::warp_any),
            WarpInstruction::Broadcast { input, id, out } => {
                writeln!(f, "auto plane_broadcast_{out} = [&](){{")?;
                writeln!(f, "    {} acc = {};", out.item(), input)?;
                writeln!(
                    f,
                    "    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{"
                )?;
                let in_optimized = input.optimized();
                let vectorization = in_optimized.item().vectorization;
                for k in 0..vectorization {
                    let acc = if vectorization == 1 {
                        "acc"
                    } else {
                        &format!("acc.i_{k}")
                    };
                    let shfl = D::warp_shuffle(acc, &format!("{id}"));
                    writeln!(f, "        {acc} = {shfl};")?;
                }
                writeln!(f, "    }};")?;
                writeln!(f, "    return acc;")?;
                writeln!(f, "}};")?;
                writeln!(f, "{} = plane_broadcast_{}();", out.fmt_left(), out)
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
    writeln!(f, "auto plane_op_{out} = [&](){{")?;
    writeln!(f, "    {} acc = {};", out.item(), input)?;
    writeln!(
        f,
        "    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{"
    )?;
    let in_optimized = input.optimized();
    let vectorization = in_optimized.item().vectorization;
    for k in 0..vectorization {
        let acc = if vectorization == 1 {
            "acc"
        } else {
            &format!("acc.i_{k}")
        };
        let shfl_xor = D::warp_shuffle_xor(acc, "offset");
        writeln!(f, "        {acc} {op} {shfl_xor};")?;
    }
    writeln!(f, "    }};")?;
    writeln!(f, "    return acc;")?;
    writeln!(f, "}};")?;
    writeln!(f, "{} = plane_op_{}();", out.fmt_left(), out)
}

fn reduce_comparison<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    cmp: &str,
) -> core::fmt::Result {
    writeln!(f, "auto plane_cmp_{out} = [&](){{")?;
    writeln!(f, "    {} acc = {};", out.item(), input)?;
    writeln!(
        f,
        "    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{"
    )?;
    let in_optimized = input.optimized();
    let instruction = match in_optimized.elem() {
        Elem::F16 | Elem::BF16 => format!("__h{cmp}"),
        Elem::F162 | Elem::BF162 => format!("__h{cmp}2"),
        _ => cmp.to_string(),
    };
    let vectorization = in_optimized.item().vectorization;
    for k in 0..vectorization {
        let acc = if vectorization == 1 {
            "acc"
        } else {
            &format!("acc.i_{k}")
        };
        let shfl_xor = D::warp_shuffle_xor(acc, "offset");
        writeln!(f, "        {acc} = {instruction}({acc}, {shfl_xor});")?;
    }
    writeln!(f, "    }};")?;
    writeln!(f, "    return acc;")?;
    writeln!(f, "}};")?;
    writeln!(f, "{} = plane_cmp_{}();", out.fmt_left(), out)
}

fn reduce_quantifier<D: Dialect, Q: Fn(&str) -> String>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    quantifier: Q,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let rhs = (0..in_optimized.item().vectorization)
        .map(|k| quantifier(&format!("{}", in_optimized.index(k))))
        .collect::<Vec<_>>()
        .join(",");
    let out_fmt = out.fmt_left();
    writeln!(f, "{out_fmt} = {{ {rhs} }};")
}
