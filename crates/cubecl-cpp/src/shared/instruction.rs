use crate::shared::FmtLeft;

use super::{
    Component, Dialect, Elem, Item, Variable, WarpInstruction, WmmaInstruction,
    barrier::BarrierOps, binary::*, pipeline::PipelineOps, unary::*,
};
use std::{
    borrow::Cow,
    fmt::{Display, Write},
    marker::PhantomData,
};

#[derive(Debug, Clone)]
pub struct BinaryInstruction<D: Dialect> {
    pub lhs: Variable<D>,
    pub rhs: Variable<D>,
    pub out: Variable<D>,
}

#[derive(Debug, Clone)]
pub struct UnaryInstruction<D: Dialect> {
    pub input: Variable<D>,
    pub out: Variable<D>,
}

#[derive(Debug, Clone)]
pub enum Instruction<D: Dialect> {
    Metadata {
        info_offset: Variable<D>,
        out: Variable<D>,
    },
    ExtendedMetadata {
        info_offset: Variable<D>,
        dim: Variable<D>,
        out: Variable<D>,
    },
    ConstLength {
        length: u32,
        out: Variable<D>,
    },
    SliceLength {
        input: Variable<D>,
        out: Variable<D>,
    },
    DeclareVariable {
        var: Variable<D>,
    },
    Modulo(BinaryInstruction<D>),
    Remainder(BinaryInstruction<D>),
    Add(BinaryInstruction<D>),
    Fma {
        a: Variable<D>,
        b: Variable<D>,
        c: Variable<D>,
        out: Variable<D>,
    },
    Div(BinaryInstruction<D>),
    Mul(BinaryInstruction<D>),
    Sub(BinaryInstruction<D>),
    Index(BinaryInstruction<D>),
    IndexAssign(BinaryInstruction<D>),
    CheckedIndex {
        len: Variable<D>,
        lhs: Variable<D>,
        rhs: Variable<D>,
        out: Variable<D>,
    },
    ConditionalRead {
        cond: Variable<D>,
        slice: Variable<D>,
        index: Variable<D>,
        fallback: Variable<D>,
        out: Variable<D>,
    },
    Assign(UnaryInstruction<D>),
    RangeLoop {
        i: Variable<D>,
        start: Variable<D>,
        end: Variable<D>,
        step: Option<Variable<D>>,
        inclusive: bool,
        instructions: Vec<Self>,
    },
    VecInit {
        inputs: Vec<Variable<D>>,
        out: Variable<D>,
    },
    Loop {
        instructions: Vec<Self>,
    },
    If {
        cond: Variable<D>,
        instructions: Vec<Self>,
    },
    IfElse {
        cond: Variable<D>,
        instructions_if: Vec<Self>,
        instructions_else: Vec<Self>,
    },
    Select {
        cond: Variable<D>,
        then: Variable<D>,
        or_else: Variable<D>,
        out: Variable<D>,
    },
    Switch {
        value: Variable<D>,
        instructions_default: Vec<Self>,
        instructions_cases: Vec<(Variable<D>, Vec<Self>)>,
    },
    Slice {
        input: Variable<D>,
        start: Variable<D>,
        end: Variable<D>,
        out: Variable<D>,
    },
    CheckedSlice {
        input: Variable<D>,
        start: Variable<D>,
        end: Variable<D>,
        out: Variable<D>,
        len: Variable<D>,
    },
    Return,
    Break,
    Equal(BinaryInstruction<D>),
    NotEqual(BinaryInstruction<D>),
    Lower(BinaryInstruction<D>),
    Greater(BinaryInstruction<D>),
    LowerEqual(BinaryInstruction<D>),
    GreaterEqual(BinaryInstruction<D>),
    Erf(UnaryInstruction<D>),
    BitwiseOr(BinaryInstruction<D>),
    BitwiseAnd(BinaryInstruction<D>),
    BitwiseXor(BinaryInstruction<D>),
    CountBits(UnaryInstruction<D>),
    ReverseBits(UnaryInstruction<D>),
    ShiftLeft(BinaryInstruction<D>),
    ShiftRight(BinaryInstruction<D>),
    BitwiseNot(UnaryInstruction<D>),
    LeadingZeros(UnaryInstruction<D>),
    FindFirstSet(UnaryInstruction<D>),
    Abs(UnaryInstruction<D>),
    Exp(UnaryInstruction<D>),
    Log(UnaryInstruction<D>),
    Log1p(UnaryInstruction<D>),
    Cos(UnaryInstruction<D>),
    Sin(UnaryInstruction<D>),
    Tanh(UnaryInstruction<D>),
    Powf(BinaryInstruction<D>),
    Sqrt(UnaryInstruction<D>),
    Min(BinaryInstruction<D>),
    Max(BinaryInstruction<D>),
    Not(UnaryInstruction<D>),
    Or(BinaryInstruction<D>),
    And(BinaryInstruction<D>),
    Clamp {
        input: Variable<D>,
        min_value: Variable<D>,
        max_value: Variable<D>,
        out: Variable<D>,
    },
    SyncThreads,
    ThreadFence,
    ProxySharedFence,
    BulkCommitGroup,
    BulkWaitGroup {
        max_pending: u32,
    },
    BulkWaitGroupRead {
        max_pending: u32,
    },
    Round(UnaryInstruction<D>),
    Ceil(UnaryInstruction<D>),
    Floor(UnaryInstruction<D>),
    Warp(WarpInstruction<D>),
    Wmma(WmmaInstruction<D>),
    Bitcast(UnaryInstruction<D>),
    AtomicLoad(UnaryInstruction<D>),
    AtomicStore(UnaryInstruction<D>),
    AtomicSwap(BinaryInstruction<D>),
    AtomicAdd(BinaryInstruction<D>),
    AtomicSub(BinaryInstruction<D>),
    AtomicMax(BinaryInstruction<D>),
    AtomicMin(BinaryInstruction<D>),
    AtomicAnd(BinaryInstruction<D>),
    AtomicOr(BinaryInstruction<D>),
    AtomicXor(BinaryInstruction<D>),
    AtomicCAS {
        input: Variable<D>,
        cmp: Variable<D>,
        val: Variable<D>,
        out: Variable<D>,
    },
    Neg(UnaryInstruction<D>),
    Magnitude(UnaryInstruction<D>),
    Normalize(UnaryInstruction<D>),
    Dot(BinaryInstruction<D>),
    Copy {
        input: Variable<D>,
        in_index: Variable<D>,
        out: Variable<D>,
        out_index: Variable<D>,
    },
    CopyBulk {
        input: Variable<D>,
        in_index: Variable<D>,
        out: Variable<D>,
        out_index: Variable<D>,
        len: u32,
    },
    Printf {
        format_string: String,
        args: Vec<Variable<D>>,
    },
    Comment {
        content: String,
    },
    Pipeline(PipelineOps<D>),
    Barrier(BarrierOps<D>),
    MemCopyAsyncTensorSharedToGlobal {
        smem_buffer: Variable<D>,
        tensor_map: Variable<D>,
        indices: Vec<Variable<D>>,
    },
    Line {
        file: Cow<'static, str>,
        line: u32,
    },
}

impl<D: Dialect> Display for Instruction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => f.write_str("return;"),
            Instruction::Break => f.write_str("break;"),
            Instruction::DeclareVariable { var } => match var {
                Variable::WmmaFragment { frag, .. } => writeln!(f, "{frag} {var};"),
                _ => {
                    let item = var.item();
                    writeln!(f, "{item} {var};")
                }
            },
            Instruction::Add(it) => Add::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Slice {
                input,
                start,
                end,
                out,
            } => {
                let item = out.item();
                writeln!(f, "const uint {out}_length = {end} - {start};")?;
                writeln!(f, "{item} *{out} = {input} + {start};")
            }
            Instruction::CheckedSlice {
                input,
                start,
                end,
                out,
                len,
            } => {
                let item = out.item();
                writeln!(f, "const uint {out}_length = min({len}, {end}) - {start};")?;
                writeln!(f, "{item} *{out} = {input} + {start};")
            }
            Instruction::Mul(it) => Mul::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Div(it) => Div::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sub(it) => Sub::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Modulo(inst) => Modulo::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::BitwiseOr(it) => BitwiseOr::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::BitwiseAnd(it) => BitwiseAnd::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::BitwiseXor(it) => BitwiseXor::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::CountBits(it) => CountBits::format(f, &it.input, &it.out),
            Instruction::ReverseBits(it) => ReverseBits::format(f, &it.input, &it.out),
            Instruction::LeadingZeros(it) => LeadingZeros::format(f, &it.input, &it.out),
            Instruction::FindFirstSet(it) => FindFirstSet::format(f, &it.input, &it.out),
            Instruction::ShiftLeft(it) => ShiftLeft::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::ShiftRight(it) => ShiftRight::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Index(it) => Index::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::IndexAssign(it) => IndexAssign::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::CheckedIndex { len, lhs, rhs, out } => {
                let item_out = out.item();
                if let Elem::Atomic(inner) = item_out.elem {
                    write!(f, "{inner}* {out} = &{lhs}[{rhs}];")
                } else {
                    let out = out.fmt_left();
                    write!(f, "{out} = ({rhs} < {len}) ? ")?;
                    Index::format_scalar(f, *lhs, *rhs, item_out)?;
                    if item_out.vectorization == 1 {
                        writeln!(f, " : {item_out}(0);")
                    } else {
                        writeln!(f, " : {item_out}{{}};")
                    }
                }
            }
            Instruction::ConditionalRead {
                cond,
                slice,
                index,
                fallback,
                out,
            } => {
                let item_fallback = fallback.item();
                let item_slice = slice.item();
                let item_out = out.item();
                let item_cond = cond.item();
                let elem_cond = item_cond.elem;

                let vf_slice = item_slice.vectorization;
                let vf_fallback = item_fallback.vectorization;
                let vf_out = item_out.vectorization;
                let vf_cond = item_cond.vectorization;

                let out = out.fmt_left();

                let should_broadcast =
                    vf_cond > 1 || item_out != item_fallback || item_out != item_slice;

                if should_broadcast {
                    let vf = usize::max(vf_cond, vf_out);
                    let vf = usize::max(vf, vf_slice);
                    let vf = usize::max(vf, vf_fallback);

                    writeln!(f, "{out} = {item_out} {{")?;
                    for i in 0..vf {
                        let fallbacki = fallback.index(i);
                        let condi = cond.index(i);
                        let condi = EnsureBoolArg {
                            var: &condi,
                            elem: &elem_cond,
                        };

                        writeln!(f, "({condi}) ? {slice}[{index} + i] : {fallbacki},")?;
                    }

                    writeln!(f, "}};")
                } else {
                    let cond = EnsureBoolArg {
                        var: &cond,
                        elem: &elem_cond,
                    };
                    writeln!(f, "{out} = ({cond}) ? {slice}[{index}] : {fallback};")
                }
            }
            Instruction::Copy {
                input,
                in_index,
                out,
                out_index,
            } => {
                writeln!(f, "{out}[{out_index}] = {input}[{in_index}];")
            }
            Instruction::CopyBulk {
                input,
                in_index,
                out,
                out_index,
                len,
            } => {
                for i in 0..*len {
                    writeln!(f, "{out}[{out_index} + {i}] = {input}[{in_index} + {i}];")?;
                }
                Ok(())
            }
            Instruction::Assign(it) => Assign::format(f, &it.input, &it.out),
            Instruction::RangeLoop {
                i,
                start,
                end,
                step,
                inclusive,
                instructions,
            } => {
                let increment = step
                    .map(|step| format!("{i} += {step}"))
                    .unwrap_or_else(|| format!("++{i}"));
                let cmp = if *inclusive { "<=" } else { "<" };
                let i_ty = i.item();

                write!(
                    f,
                    "
for ({i_ty} {i} = {start}; {i} {cmp} {end}; {increment}) {{
"
                )?;
                for instruction in instructions {
                    write!(f, "{instruction}")?;
                }

                f.write_str("}\n")
            }
            Instruction::Loop { instructions } => {
                writeln!(f, "while (true) {{")?;
                for i in instructions {
                    write!(f, "{i}")?;
                }
                f.write_str("}\n")
            }
            Instruction::If { cond, instructions } => {
                writeln!(f, "if ({cond}) {{")?;
                for i in instructions {
                    write!(f, "{i}")?;
                }
                f.write_str("}\n")
            }
            Instruction::IfElse {
                cond,
                instructions_if,
                instructions_else,
            } => {
                writeln!(f, "if ({cond}) {{")?;
                for i in instructions_if {
                    write!(f, "{i}")?;
                }
                f.write_str("} else {\n")?;
                for i in instructions_else {
                    write!(f, "{i}")?;
                }
                f.write_str("}\n")
            }
            Instruction::Select {
                cond,
                then,
                or_else,
                out,
            } => {
                let item_or_else = or_else.item();
                let item_then = then.item();
                let item_out = out.item();

                let vf_then = item_then.vectorization;
                let vf_or_else = item_or_else.vectorization;
                let vf_out = item_out.vectorization;
                let vf_cond = cond.item().vectorization;

                let item_out = out.item();
                let cond_elem = cond.item().elem;
                let out = out.fmt_left();

                let should_broadcast =
                    vf_cond > 1 || item_out != item_or_else || item_out != item_then;

                if should_broadcast {
                    let vf = usize::max(vf_cond, vf_out);
                    let vf = usize::max(vf, vf_then);
                    let vf = usize::max(vf, vf_or_else);

                    writeln!(f, "{out} = {item_out} {{")?;
                    for i in 0..vf {
                        let theni = then.index(i);
                        let or_elsei = or_else.index(i);
                        let condi = cond.index(i);
                        let condi = EnsureBoolArg {
                            var: &condi,
                            elem: &cond_elem,
                        };

                        writeln!(f, "({condi}) ? {theni} : {or_elsei},")?;
                    }

                    writeln!(f, "}};")
                } else {
                    let cond = EnsureBoolArg {
                        var: &cond,
                        elem: &cond_elem,
                    };
                    writeln!(f, "{out} = ({cond}) ? {then} : {or_else};")
                }
            }
            Instruction::Switch {
                value,
                instructions_default,
                instructions_cases,
            } => {
                writeln!(f, "switch({value}) {{")?;
                for (value, block) in instructions_cases {
                    write!(f, "case {value}:\n{{\n")?;
                    for i in block {
                        i.fmt(f)?;
                    }
                    f.write_str("break;\n}\n")?;
                }
                f.write_str("default:\n{")?;
                for i in instructions_default {
                    i.fmt(f)?;
                }
                f.write_str("}\n}\n")
            }
            Instruction::Metadata { info_offset, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[{info_offset}];")
            }
            Instruction::ExtendedMetadata {
                info_offset,
                dim,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[info[{info_offset}] + {dim}];")
            }
            Instruction::Equal(it) => Equal::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::NotEqual(it) => NotEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Lower(it) => Lower::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Greater(it) => Greater::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::LowerEqual(it) => LowerEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::GreaterEqual(it) => GreaterEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Erf(it) => Erf::format(f, &it.input, &it.out),
            Instruction::Abs(it) => Abs::format(f, &it.input, &it.out),
            Instruction::Exp(it) => Exp::format(f, &it.input, &it.out),
            Instruction::Log(it) => Log::format(f, &it.input, &it.out),
            Instruction::Log1p(it) => Log1p::format(f, &it.input, &it.out),
            Instruction::Cos(it) => Cos::format(f, &it.input, &it.out),
            Instruction::Sin(it) => Sin::format(f, &it.input, &it.out),
            Instruction::Tanh(it) => Tanh::format(f, &it.input, &it.out),
            Instruction::Powf(it) => Powf::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sqrt(it) => Sqrt::format(f, &it.input, &it.out),
            Instruction::Max(it) => Max::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Min(it) => Min::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Not(it) => Not::format(f, &it.input, &it.out),
            Instruction::BitwiseNot(it) => BitwiseNot::format(f, &it.input, &it.out),
            Instruction::Or(it) => Or::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::And(it) => And::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => Clamp::format(f, input, min_value, max_value, out),
            Instruction::SyncThreads => f.write_str("__syncthreads();\n"),
            Instruction::ThreadFence => f.write_str("__threadfence();\n"),
            Instruction::Round(it) => Round::format(f, &it.input, &it.out),
            Instruction::Ceil(it) => Ceil::format(f, &it.input, &it.out),
            Instruction::Floor(it) => Floor::format(f, &it.input, &it.out),
            Instruction::SliceLength { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = {input}_length;")
            }
            Instruction::ConstLength { length, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = {length};")
            }
            Instruction::Warp(it) => write!(f, "{it}"),
            Instruction::Fma { a, b, c, out } => Fma::format(f, a, b, c, out),
            Instruction::Wmma(it) => write!(f, "{it}"),
            Instruction::Bitcast(UnaryInstruction { input, out }) => {
                let qualifier = out.const_qualifier();
                let out_elem = out.elem();
                let out = out.fmt_left();

                match (input.elem(), out_elem) {
                    (Elem::F32, Elem::I32) => {
                        writeln!(f, "{out} = __float_as_int({input});")
                    }
                    (Elem::F32, Elem::U32) => {
                        writeln!(f, "{out} = __float_as_uint({input});")
                    }
                    (Elem::F16, Elem::I32) => {
                        writeln!(f, "{out} = __half_as_short({input});")
                    }
                    (Elem::F16, Elem::U32) => {
                        writeln!(f, "{out} = __half_as_ushort({input});")
                    }
                    (Elem::BF16, Elem::I32) => {
                        writeln!(f, "{out} = __bfloat16_as_short({input});")
                    }
                    (Elem::BF16, Elem::U32) => {
                        writeln!(f, "{out} = __bfloat16_as_ushort({input});")
                    }
                    (Elem::I32, Elem::F32) => {
                        writeln!(f, "{out} = __int_as_float({input});")
                    }
                    (Elem::I32, Elem::F16) => {
                        writeln!(f, "{out} = __short_as_half({input});")
                    }
                    (Elem::I32, Elem::BF16) => {
                        writeln!(f, "{out} = __short_as_bfloat16({input});")
                    }
                    (Elem::U32, Elem::F32) => {
                        writeln!(f, "{out} = __uint_as_float({input});")
                    }
                    (Elem::U32, Elem::F16) => {
                        writeln!(f, "{out} = __ushort_as_half({input});")
                    }
                    (Elem::U32, Elem::BF16) => {
                        writeln!(f, "{out} = __ushort_as_bfloat16({input});")
                    }
                    (Elem::I32, Elem::U32) => {
                        writeln!(f, "{out} = reinterpret_cast<uint{qualifier}&>({input});")
                    }
                    elem => panic!("Unsupported type for bitcasting {elem:?}"),
                }
            }
            Instruction::AtomicCAS {
                input,
                cmp,
                val,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicCAS({input}, {cmp}, {val});")
            }
            Instruction::AtomicSwap(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicExch({lhs}, {rhs});")
            }
            Instruction::AtomicAdd(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                match rhs.elem() {
                    Elem::I64 => {
                        writeln!(
                            f,
                            "{out} = atomicAdd(reinterpret_cast<{uint}*>({lhs}), {uint}({rhs}));",
                            uint = Elem::<D>::U64
                        )
                    }
                    _ => writeln!(f, "{out} = atomicAdd({lhs}, {rhs});"),
                }
            }
            Instruction::AtomicSub(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                match rhs.elem() {
                    Elem::U32 | Elem::I32 => {
                        writeln!(f, "{out} = atomicSub({lhs}, {rhs});")
                    }
                    Elem::U64 => {
                        writeln!(f, "{out} = atomicAdd({lhs}, -{rhs});",)
                    }
                    Elem::I64 => {
                        writeln!(
                            f,
                            "{out} = atomicAdd(reinterpret_cast<{uint}*>({lhs}), {uint}(-{rhs}));",
                            uint = Elem::<D>::U64
                        )
                    }
                    _ => writeln!(f, "{out} = atomicAdd({lhs}, -{rhs});"),
                }
            }
            Instruction::AtomicMax(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicMax({lhs}, {rhs});")
            }
            Instruction::AtomicMin(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicMin({lhs}, {rhs});")
            }
            Instruction::AtomicAnd(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicAnd({lhs}, {rhs});")
            }
            Instruction::AtomicOr(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicOr({lhs}, {rhs});")
            }
            Instruction::AtomicXor(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicXor({lhs}, {rhs});")
            }
            Instruction::AtomicLoad(UnaryInstruction { input, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicAdd({input}, 0);")
            }
            Instruction::AtomicStore(UnaryInstruction { input, out }) => {
                writeln!(f, "atomicExch({out}, {input});")
            }
            Instruction::Remainder(inst) => Remainder::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::Neg(UnaryInstruction { input, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = -{input};")
            }
            Instruction::Normalize(inst) => Normalize::format(f, &inst.input, &inst.out),
            Instruction::Magnitude(inst) => Magnitude::format(f, &inst.input, &inst.out),
            Instruction::Dot(inst) => Dot::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::VecInit { inputs, out } => {
                let item = out.item();
                let inputs = inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                let out = out.fmt_left();
                writeln!(f, "{out} = {item}{{{}}};", inputs.join(","))
            }
            Instruction::Printf {
                format_string,
                args,
            } => {
                let format_string = escape_string(format_string);
                let args = args.iter().map(|arg| format!("{arg}")).collect::<Vec<_>>();
                let args = match args.is_empty() {
                    true => "".to_string(),
                    false => format!(", {}", args.join(",")),
                };
                writeln!(f, "printf(\"{format_string}\"{args});")
            }
            Instruction::Comment { content } => {
                if content.contains('\n') {
                    writeln!(f, "/* {content} */")
                } else {
                    writeln!(f, "// {content}")
                }
            }
            Instruction::Pipeline(pipeline_ops) => write!(f, "{pipeline_ops}"),
            Instruction::Barrier(barrier_ops) => write!(f, "{barrier_ops}"),
            Instruction::Line { file, line } => writeln!(f, "#line {line} \"{file}\""),
            Instruction::ProxySharedFence => {
                writeln!(
                    f,
                    "cuda::device::experimental::fence_proxy_async_shared_cta();"
                )
            }
            Instruction::BulkCommitGroup => writeln!(
                f,
                "cuda::device::experimental::cp_async_bulk_commit_group();"
            ),
            Instruction::BulkWaitGroup { max_pending } => writeln!(
                f,
                "cuda::device::experimental::cp_async_bulk_wait_group<{max_pending}>();"
            ),
            Instruction::BulkWaitGroupRead { max_pending } => writeln!(
                f,
                "cuda::device::experimental::cp_async_bulk_wait_group_read<{max_pending}>();"
            ),
            Instruction::MemCopyAsyncTensorSharedToGlobal {
                smem_buffer,
                tensor_map,
                indices,
            } => {
                let rank = indices.len();
                let indices = indices.iter().rev().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                writeln!(
                    f,
                    "cuda::device::experimental::cp_async_bulk_tensor_{rank}d_shared_to_global(&{tensor_map}, {indices} &{smem_buffer});"
                )
            }
        }
    }
}

fn escape_string(format_string: &str) -> String {
    format_string
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
}

struct Fma<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Fma<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        a: &Variable<D>,
        b: &Variable<D>,
        c: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let out_item = out.item();
        let num = out_item.vectorization;

        let out = out.fmt_left();
        if num == 1 {
            writeln!(f, "{out} = fma({a}, {b}, {c});")
        } else {
            writeln!(f, "{out} = {out_item}{{")?;

            for i in 0..num {
                let ai = a.index(i);
                let bi = b.index(i);
                let ci = c.index(i);

                writeln!(f, "fma({ai}, {bi}, {ci}),")?;
            }
            f.write_str("};\n")
        }
    }
}

struct Clamp<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Clamp<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable<D>,
        min_value: &Variable<D>,
        max_value: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let input = input.optimized();
        let min_value = min_value.optimized();
        let max_value = max_value.optimized();
        let out = out.optimized();
        let out_item = out.item();
        let num = out_item.vectorization;

        let (min, max) = match out.elem() {
            Elem::F16 | Elem::BF16 => ("__hmin", "__hmax"),
            Elem::F162 | Elem::BF162 => ("__hmin2", "__hmax2"),
            _ => ("min", "max"),
        };

        let out = out.fmt_left();
        if num == 1 {
            writeln!(
                f,
                "{out} = {max}({min_value}, {min}({max_value}, {input}));"
            )
        } else {
            writeln!(f, "{out} = {out_item}{{")?;
            for i in 0..num {
                let inputi = input.index(i);
                let mini = min_value.index(i);
                let maxi = max_value.index(i);

                writeln!(f, "{max}({mini}, {min}({maxi}, {inputi})),")?;
            }

            f.write_str("};\n")
        }
    }
}

struct Remainder<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Remainder<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let floor = |elem| match elem {
            Elem::F16 | Elem::BF16 => "hfloor",
            Elem::F162 | Elem::BF162 => "h2floor",
            _ => "floor",
        };

        if out.item().vectorization == 1 {
            let floor = floor(out.elem());

            let out = out.fmt_left();
            return writeln!(f, "{out} = {lhs} - {rhs} * {floor}({lhs} / {rhs});");
        }

        let optimized = Variable::optimized_args([*lhs, *rhs, *out]);
        let [lhs, rhs, out_optimized] = optimized.args;

        let item_out_original = out.item();
        let item_out_optimized = out_optimized.item();

        let index = match optimized.optimization_factor {
            Some(factor) => item_out_original.vectorization / factor,
            None => item_out_optimized.vectorization,
        };

        let floor = floor(*item_out_optimized.elem());

        let mut write_op =
            |lhs: &Variable<D>, rhs: &Variable<D>, out: &Variable<D>, item_out: Item<D>| {
                let out = out.fmt_left();
                writeln!(f, "{out} = {item_out}{{")?;
                for i in 0..index {
                    let lhsi = lhs.index(i);
                    let rhsi = rhs.index(i);

                    writeln!(f, "{lhsi} - {rhsi} * {floor}({lhsi} / {rhsi})")?;
                    f.write_str(", ")?;
                }

                f.write_str("};\n")
            };

        if item_out_original == item_out_optimized {
            write_op(&lhs, &rhs, out, item_out_optimized)
        } else {
            let out_tmp = Variable::tmp(item_out_optimized);

            write_op(&lhs, &rhs, &out_tmp, item_out_optimized)?;

            let qualifier = out.const_qualifier();
            let out = out.fmt_left();

            writeln!(
                f,
                "{out} = reinterpret_cast<{item_out_original}{qualifier}&>({out_tmp});\n"
            )?;

            Ok(())
        }
    }
}

struct Magnitude<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Magnitude<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let num = input.item().vectorization;
        let elem = input.elem();

        let mag = format!("{out}_mag");

        writeln!(f, "{} {mag} = 0.0;", out.item())?;

        for i in 0..num {
            let input_i = input.index(i);
            writeln!(f, "{mag} += {input_i} * {input_i};")?;
        }

        let out = out.fmt_left();
        write!(f, "{out} = ")?;
        Sqrt::format_unary(f, &mag, elem)?;
        f.write_str(";\n")
    }
}

struct Normalize<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Normalize<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let num = input.item().vectorization;
        let elem = input.elem();
        let norm = format!("{out}_norm");

        let out_item = out.item();
        let out = out.fmt_left();
        writeln!(f, "{elem} {norm} = 0.0;")?;

        for i in 0..num {
            let input_i = input.index(i);
            writeln!(f, "{norm} += {input_i} * {input_i};")?;
        }

        write!(f, "{norm} = ")?;
        Sqrt::format_unary(f, &norm, elem)?;
        f.write_str(";\n")?;

        if num == 1 {
            writeln!(f, "{out} = {input} / {norm};")
        } else {
            write!(f, "{out} = {out_item}{{")?;
            for i in 0..num {
                let input_i = input.index(i);

                writeln!(f, "{input_i} / {norm},")?;
            }

            f.write_str("};\n")
        }
    }
}

struct Dot<D: Dialect> {
    _dialect: PhantomData<D>,
}

impl<D: Dialect> Dot<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let num = lhs.item().vectorization;

        let muls = (0..num)
            .map(|i| {
                let lhs_i = lhs.index(i);
                let rhs_i = rhs.index(i);
                format!("{lhs_i} * {rhs_i}")
            })
            .collect::<Vec<_>>();

        let out = out.fmt_left();
        writeln!(f, "{out} = {};", muls.join(" + "))
    }
}

struct EnsureBoolArg<'a, V: Display, D: Dialect> {
    var: &'a V,
    elem: &'a Elem<D>,
}

impl<V: Display, D: Dialect> Display for EnsureBoolArg<'_, V, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elem != &Elem::Bool {
            write!(f, "bool({})", self.var)
        } else {
            write!(f, "{}", self.var)
        }
    }
}
