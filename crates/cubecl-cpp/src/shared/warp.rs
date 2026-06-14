use std::fmt::Display;

use crate::shared::{Builtin, Component, FmtLeft};

use super::{Dialect, IndexedValue, Item, Value};

#[derive(Clone, Debug)]
pub enum WarpInstruction<D: Dialect> {
    ReduceSum {
        input: Value<D>,
        out: Value<D>,
    },
    InclusiveSum {
        input: Value<D>,
        out: Value<D>,
    },
    ExclusiveSum {
        input: Value<D>,
        out: Value<D>,
    },
    ReduceProd {
        input: Value<D>,
        out: Value<D>,
    },
    InclusiveProd {
        input: Value<D>,
        out: Value<D>,
    },
    ExclusiveProd {
        input: Value<D>,
        out: Value<D>,
    },
    ReduceMax {
        input: Value<D>,
        out: Value<D>,
    },
    ReduceMin {
        input: Value<D>,
        out: Value<D>,
    },
    ElectFallback {
        out: Value<D>,
    },
    Elect {
        out: Value<D>,
    },
    All {
        input: Value<D>,
        out: Value<D>,
    },
    Any {
        input: Value<D>,
        out: Value<D>,
    },
    Ballot {
        input: Value<D>,
        out: Value<D>,
    },
    Broadcast {
        input: Value<D>,
        id: Value<D>,
        out: Value<D>,
    },
    Shuffle {
        input: Value<D>,
        src_lane: Value<D>,
        out: Value<D>,
    },
    ShuffleXor {
        input: Value<D>,
        mask: Value<D>,
        out: Value<D>,
    },
    ShuffleUp {
        input: Value<D>,
        delta: Value<D>,
        out: Value<D>,
    },
    ShuffleDown {
        input: Value<D>,
        delta: Value<D>,
        out: Value<D>,
    },
}

impl<D: Dialect> Display for WarpInstruction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarpInstruction::ReduceSum { input, out } => D::warp_reduce_sum(f, input, out),
            WarpInstruction::ReduceProd { input, out } => D::warp_reduce_prod(f, input, out),
            WarpInstruction::ReduceMax { input, out } => D::warp_reduce_max(f, input, out),
            WarpInstruction::ReduceMin { input, out } => D::warp_reduce_min(f, input, out),
            WarpInstruction::All { input, out } => D::warp_reduce_all(f, input, out),
            WarpInstruction::Any { input, out } => D::warp_reduce_any(f, input, out),

            WarpInstruction::InclusiveSum { input, out } => {
                D::warp_reduce_sum_inclusive(f, input, out)
            }
            WarpInstruction::InclusiveProd { input, out } => {
                D::warp_reduce_prod_inclusive(f, input, out)
            }
            WarpInstruction::ExclusiveSum { input, out } => {
                D::warp_reduce_sum_exclusive(f, input, out)
            }
            WarpInstruction::ExclusiveProd { input, out } => {
                D::warp_reduce_prod_exclusive(f, input, out)
            }
            WarpInstruction::Ballot { input, out } => {
                assert_eq!(
                    input.item().vectorization(),
                    1,
                    "Ballot can't support vectorized input"
                );
                let out_fmt = out.fmt_left();
                write!(
                    f,
                    "
{out_fmt} = {{ "
                )?;
                D::compile_warp_ballot(f, input, out.item().elem())?;
                writeln!(f, ", 0, 0, 0 }};")
            }
            WarpInstruction::Broadcast { input, id, out } => reduce_broadcast(f, input, out, id),
            WarpInstruction::Shuffle {
                input,
                src_lane,
                out,
            } => {
                let out_fmt = out.fmt_left();
                write!(f, "{out_fmt} = {{ ")?;
                for i in 0..input.item().vectorization() {
                    let comma = if i > 0 { ", " } else { "" };
                    write!(f, "{comma}")?;
                    D::compile_warp_shuffle(
                        f,
                        &format!("{}", input.index(i)),
                        input.item().elem(),
                        &format!("{src_lane}"),
                    )?;
                }
                writeln!(f, " }};")
            }
            WarpInstruction::ShuffleXor { input, mask, out } => {
                let out_fmt = out.fmt_left();
                write!(f, "{out_fmt} = {{ ")?;
                for i in 0..input.item().vectorization() {
                    let comma = if i > 0 { ", " } else { "" };
                    write!(f, "{comma}")?;
                    D::compile_warp_shuffle_xor(
                        f,
                        &format!("{}", input.index(i)),
                        input.item().elem(),
                        &format!("{mask}"),
                    )?;
                }
                writeln!(f, " }};")
            }
            WarpInstruction::ShuffleUp { input, delta, out } => {
                let out_fmt = out.fmt_left();
                write!(f, "{out_fmt} = {{ ")?;
                for i in 0..input.item().vectorization() {
                    let comma = if i > 0 { ", " } else { "" };
                    write!(f, "{comma}")?;
                    D::compile_warp_shuffle_up(
                        f,
                        &format!("{}", input.index(i)),
                        input.item().elem(),
                        &format!("{delta}"),
                    )?;
                }
                writeln!(f, " }};")
            }
            WarpInstruction::ShuffleDown { input, delta, out } => {
                let out_fmt = out.fmt_left();
                write!(f, "{out_fmt} = {{ ")?;
                for i in 0..input.item().vectorization() {
                    let comma = if i > 0 { ", " } else { "" };
                    write!(f, "{comma}")?;
                    D::compile_warp_shuffle_down(
                        f,
                        &format!("{}", input.index(i)),
                        input.item().elem(),
                        &format!("{delta}"),
                    )?;
                }
                writeln!(f, " }};")
            }
            WarpInstruction::ElectFallback { out } => {
                let out = out.fmt_left();
                write!(
                    f,
                    "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
                )
            }
            WarpInstruction::Elect { out } => {
                let out = out.fmt_left();
                D::compile_warp_elect(f, &out)
            }
        }
    }
}

pub(crate) fn reduce_operator<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    op: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    reduce_with_loop(f, input, out, acc_item, |f, acc, index| {
        let acc_indexed = maybe_index(acc, index);
        write!(f, "{acc_indexed} {op} ")?;
        D::compile_warp_shuffle_xor(f, &acc_indexed, acc.item().elem(), "offset")?;
        writeln!(f, ";")
    })
}

pub(crate) fn reduce_comparison<
    D: Dialect,
    I: Fn(&mut core::fmt::Formatter<'_>, Item<D>) -> std::fmt::Result,
>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    instruction: I,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();
    reduce_with_loop(f, input, out, acc_item, |f, acc, index| {
        let acc_indexed = maybe_index(acc, index);
        let acc_elem = acc_item.elem();
        write!(f, "        {acc_indexed} = ")?;
        instruction(f, in_optimized.item())?;
        write!(f, "({acc_indexed}, ")?;
        D::compile_warp_shuffle_xor(f, &acc_indexed, acc_elem, "offset")?;
        writeln!(f, ");")
    })
}

pub(crate) fn reduce_inclusive<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    op: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    reduce_with_loop(f, input, out, acc_item, |f, acc, index| {
        let acc_indexed = maybe_index(acc, index);
        let tmp = Value::tmp(Item::Scalar(*acc_item.elem()));
        let tmp_left = tmp.fmt_left();
        let lane_id = Builtin::<D>::UnitPosPlane;
        write!(
            f,
            "
{tmp_left} = "
        )?;
        D::compile_warp_shuffle_up(f, &acc_indexed, acc_item.elem(), "offset")?;
        write!(
            f,
            ";
if({lane_id} >= offset) {{
    {acc_indexed} {op} {tmp};
}}
"
        )
    })
}

pub(crate) fn reduce_exclusive<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    op: &str,
    default: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    let inclusive = Value::tmp(acc_item);
    reduce_inclusive(f, input, &inclusive, op)?;
    let shfl = Value::tmp(acc_item);
    writeln!(f, "{} = {{", shfl.fmt_left())?;
    for k in 0..acc_item.vectorization() {
        let inclusive_indexed = maybe_index(&inclusive, k);
        let comma = if k > 0 { ", " } else { "" };
        write!(f, "{comma}")?;
        D::compile_warp_shuffle_up(f, &inclusive_indexed.to_string(), acc_item.elem(), "1")?;
    }
    writeln!(f, "}};")?;
    let lane_id = Builtin::<D>::UnitPosPlane;

    write!(
        f,
        "{} = ({lane_id} == 0) ? {}{{",
        out.fmt_left(),
        out.item(),
    )?;
    for _ in 0..out.item().vectorization() {
        write!(f, "{default},")?;
    }
    writeln!(f, "}} : {};", cast(&shfl, out.item()))
}

pub(crate) fn reduce_broadcast<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    id: &Value<D>,
) -> core::fmt::Result {
    let out_fmt = out.fmt_left();
    write!(f, "{out_fmt} = {{ ")?;
    for i in 0..input.item().vectorization() {
        let comma = if i > 0 { ", " } else { "" };
        write!(f, "{comma}")?;
        D::compile_warp_shuffle(
            f,
            &format!("{}", input.index(i)),
            input.item().elem(),
            &format!("{id}"),
        )?;
    }
    writeln!(f, " }};")
}

fn reduce_with_loop<
    D: Dialect,
    I: Fn(&mut core::fmt::Formatter<'_>, &Value<D>, usize) -> std::fmt::Result,
>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    acc_item: Item<D>,
    instruction: I,
) -> core::fmt::Result {
    let acc = Value::tmp(acc_item);
    let vectorization = acc_item.vectorization();

    writeln!(f, "auto plane_{out} = [&]() -> {} {{", out.item())?;
    writeln!(f, "    {} {} = {};", acc_item, acc, cast(input, acc_item))?;
    write!(f, "    for (uint offset = 1; offset < ")?;
    D::compile_plane_dim_checked(f)?;
    writeln!(f, "; offset *=2 ) {{")?;
    for k in 0..vectorization {
        instruction(f, &acc, k)?;
    }
    writeln!(f, "    }};")?;
    writeln!(f, "    return {};", cast(&acc, out.item()))?;
    writeln!(f, "}};")?;
    writeln!(f, "{} = plane_{}();", out.fmt_left(), out)
}

pub(crate) fn reduce_quantifier<
    D: Dialect,
    Q: Fn(&mut core::fmt::Formatter<'_>, &IndexedValue<D>) -> std::fmt::Result,
>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Value<D>,
    out: &Value<D>,
    quantifier: Q,
) -> core::fmt::Result {
    let out_fmt = out.fmt_left();
    write!(f, "{out_fmt} = {{ ")?;
    for i in 0..input.item().vectorization() {
        let comma = if i > 0 { ", " } else { "" };
        write!(f, "{comma}")?;
        quantifier(f, &input.index(i))?;
    }
    writeln!(f, "}};")
}

fn cast<D: Dialect>(input: &Value<D>, target: Item<D>) -> String {
    if target != input.item() {
        let addr_space = D::address_space_for_value(input);
        let qualifier = input.const_qualifier();
        format!("reinterpret_cast<{addr_space}{target}{qualifier}&>({input})")
    } else {
        format!("{input}")
    }
}

fn maybe_index<D: Dialect>(val: &Value<D>, k: usize) -> String {
    if val.item().vectorization() > 1 {
        format!("{val}.i_{k}")
    } else {
        format!("{val}")
    }
}
