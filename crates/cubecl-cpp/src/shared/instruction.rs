use crate::shared::{FP8Kind, FmtLeft};

use super::{
    Component, Dialect, Elem, Item, Variable, WarpInstruction, WmmaInstruction,
    barrier::BarrierOps, binary::*, pipeline::PipelineOps, unary::*,
};
use core::fmt;
use std::{
    borrow::Cow,
    fmt::{Display, Write},
    marker::PhantomData,
};

pub(crate) const INFO_NAME: &str = "info";
pub(crate) const STATIC_INFO_NAME: &str = "static_info";

#[derive(Debug, Clone)]
pub struct BinaryInstruction<D: Dialect> {
    pub lhs: Variable<D>,
    pub rhs: Variable<D>,
    pub out: Variable<D>,
}

#[derive(Debug, Clone)]
pub struct IndexInstruction<D: Dialect> {
    pub list: Variable<D>,
    pub index: Variable<D>,
    pub line_size: u32,
    pub out: Variable<D>,
}

#[derive(Debug, Clone)]
pub struct IndexAssignInstruction<D: Dialect> {
    pub index: Variable<D>,
    pub value: Variable<D>,
    pub line_size: u32,
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
        split_meta: bool,
        out: Variable<D>,
    },
    ExtendedMetadata {
        info_offset: Variable<D>,
        dim: Variable<D>,
        split_meta: bool,
        static_offset: u32,
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
    HiMul(BinaryInstruction<D>),
    Index(IndexInstruction<D>),
    IndexAssign(IndexAssignInstruction<D>),
    Assign(UnaryInstruction<D>),
    SpecialCast(UnaryInstruction<D>),
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
    ReinterpretSlice {
        input: Variable<D>,
        line_size: u32,
        out: Variable<D>,
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
        smem_offset: Variable<D>,
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
                let addr_space = D::address_space_for_variable(input);
                writeln!(f, "const uint {out}_length = {end} - {start};")?;
                writeln!(f, "{addr_space}{item} *{out} = {input} + {start};")
            }
            Instruction::CheckedSlice {
                input,
                start,
                end,
                out,
                len,
            } => {
                let item = out.item();
                let addr_space = D::address_space_for_variable(input);
                writeln!(f, "const uint {out}_length = min({len}, {end}) - {start};")?;
                writeln!(f, "{addr_space}{item} *{out} = {input} + {start};")
            }
            Instruction::ReinterpretSlice {
                input,
                line_size,
                out,
            } => {
                let mut item = out.item();
                item.vectorization = *line_size as usize;
                let addr_space = D::address_space_for_variable(input);

                writeln!(
                    f,
                    "{addr_space}{item} *{out} = reinterpret_cast<{item}*>({input});"
                )
            }
            Instruction::Mul(it) => Mul::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Div(it) => Div::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sub(it) => Sub::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::HiMul(it) => HiMul::format(f, &it.lhs, &it.rhs, &it.out),
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
            Instruction::Index(it) => Index::format(f, &it.list, &it.index, &it.out, it.line_size),
            Instruction::IndexAssign(it) => {
                IndexAssign::format(f, &it.index, &it.value, &it.out, it.line_size)
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
            Instruction::Metadata {
                info_offset,
                split_meta,
                out,
            } => {
                let out = out.fmt_left();
                match *split_meta {
                    true => writeln!(f, "{out} = static_info.x[{info_offset}];"),
                    false => writeln!(f, "{out} = {INFO_NAME}[{info_offset}];"),
                }
            }
            Instruction::ExtendedMetadata {
                info_offset,
                dim,
                split_meta,
                static_offset,
                out,
            } => {
                let out = out.fmt_left();
                match *split_meta {
                    true => writeln!(
                        f,
                        "{out} = {INFO_NAME}[{STATIC_INFO_NAME}.x[{info_offset}] + {dim} - {static_offset}];"
                    ),
                    false => writeln!(
                        f,
                        "{out} = {INFO_NAME}[{INFO_NAME}[{info_offset}] + {dim}];"
                    ),
                }
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
            Instruction::SyncThreads => D::compile_instruction_sync_threads(f),
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
                let input_item = input.item();
                let out_item = out.item();

                if out_item.elem.size() * out_item.vectorization
                    != input.item().elem.size() * input.item().vectorization
                {
                    panic!("Unsupported type for bitcasting {out_item:?} from {input_item:?}");
                } else {
                    let out = out.fmt_left();
                    let addr_space = D::address_space_for_variable(input);
                    writeln!(
                        f,
                        "{out} = reinterpret_cast<{addr_space}{out_item}{qualifier}&>({input});"
                    )
                }
            }
            Instruction::AtomicAdd(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_add(f, lhs, rhs, out)
            }
            Instruction::AtomicAnd(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_and(f, lhs, rhs, out)
            }
            Instruction::AtomicCAS {
                input,
                cmp,
                val,
                out,
            } => D::compile_atomic_cas(f, input, cmp, val, out),
            Instruction::AtomicLoad(UnaryInstruction { input, out }) => {
                D::compile_atomic_load(f, input, out)
            }
            Instruction::AtomicMax(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_max(f, lhs, rhs, out)
            }
            Instruction::AtomicMin(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_min(f, lhs, rhs, out)
            }
            Instruction::AtomicOr(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_or(f, lhs, rhs, out)
            }
            Instruction::AtomicStore(UnaryInstruction { input, out }) => {
                D::compile_atomic_store(f, input, out)
            }
            Instruction::AtomicSub(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_sub(f, lhs, rhs, out)
            }
            Instruction::AtomicSwap(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_swap(f, lhs, rhs, out)
            }
            Instruction::AtomicXor(BinaryInstruction { lhs, rhs, out }) => {
                D::compile_atomic_xor(f, lhs, rhs, out)
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
            } => D::compile_instruction_printf(f, format_string, args),
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
                smem_offset,
                tensor_map,
                indices,
            } => {
                let rank = indices.len();
                let smem_ptr = smem_buffer.fmt_ptr();
                let indices = indices.iter().rev().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                writeln!(
                    f,
                    "cuda::device::experimental::cp_async_bulk_tensor_{rank}d_shared_to_global(&{tensor_map}, {indices} {smem_ptr} + {smem_offset});"
                )
            }
            Instruction::SpecialCast(UnaryInstruction { input, out }) => {
                // Only supported in CUDA so I'm putting it here. Move to dialect if necessary.
                special_cast::<D>(f, input, out)
            }
        }
    }
}

/// special cast function for recursive conversion in the case of minifloat to minifloat conversion
fn special_cast<D: Dialect>(
    f: &mut std::fmt::Formatter,
    input: &Variable<D>,
    out: &Variable<D>,
) -> fmt::Result {
    let mut current_in = *input;

    if matches!(input.elem(), Elem::FP4(_) | Elem::FP6(_) | Elem::FP8(_)) {
        let mut item = out.item();
        item.elem = match input.elem() {
            Elem::FP8(FP8Kind::UE8M0) => Elem::BF16,
            _ => Elem::F16,
        };
        let out_var = if item == out.item() {
            *out
        } else {
            Variable::tmp(item)
        };
        if item.elem == Elem::F16 {
            cast_minifloat_to_half(f, current_in, out_var)?;
        } else {
            cast_scale_to_bfloat(f, current_in, out_var)?;
        }
        current_in = out_var;
    }

    if matches!(
        current_in.elem(),
        Elem::U8
            | Elem::U16
            | Elem::U32
            | Elem::U64
            | Elem::I8
            | Elem::I16
            | Elem::I32
            | Elem::I64
            | Elem::Bool
    ) {
        // Precision is irrelevant for int, so use bf16 for the range
        let tmp = Variable::tmp(Item {
            elem: Elem::BF16,
            vectorization: input.item().vectorization,
            native: input.item().native,
        });
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: tmp,
        });
        writeln!(f, "{assign}")?;
        current_in = tmp;
    }

    if matches!(out.elem(), Elem::FP4(_) | Elem::FP6(_)) {
        return cast_to_fp4_fp6(f, current_in, *out);
    }

    if matches!(out.elem(), Elem::FP8(FP8Kind::UE8M0)) {
        // Scale can't be converted from half...
        if matches!(current_in.elem(), Elem::F16) {
            let mut item = current_in.item();
            item.elem = Elem::BF16;
            let tmp = Variable::tmp(item);
            let assign = Instruction::Assign(UnaryInstruction {
                input: current_in,
                out: tmp,
            });
            writeln!(f, "{assign}")?;
            current_in = tmp;
        }
        return cast_to_scale(f, current_in, *out);
    }

    if matches!(out.elem(), Elem::FP8(_)) {
        return cast_to_fp8(f, current_in, *out);
    }

    if current_in.item() != out.item() {
        let assign = Instruction::Assign(UnaryInstruction {
            input: current_in,
            out: *out,
        });
        writeln!(f, "{assign}")?;
    }

    Ok(())
}

fn cast_to_fp4_fp6<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out.item().vectorization / out_opt.item().vectorization;
    let packed = packing > 1;
    let pack_suffix = if packed { "2" } else { "" };

    let (out_ty, interpretation) = match out_opt.elem() {
        Elem::FP4(kind) => ("fp4", format!("{kind:?}")),
        Elem::FP4x2(kind) => ("fp4x2", format!("{kind:?}")),
        Elem::FP6(kind) => ("fp6", format!("{kind:?}")),
        Elem::FP6x2(kind) => ("fp6x2", format!("{kind:?}")),
        _ => unreachable!("Must be fp4 or fp6"),
    };

    let in_ty = match input.elem() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::F16 => format!("halfraw{pack_suffix}"),
        Elem::BF16 => format!("bfloat16raw{pack_suffix}"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_{interpretation}, cudaRoundNearest)",
        )
    })
}

fn cast_to_scale<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out.item().vectorization / out_opt.item().vectorization;
    let packed = packing > 1;
    let pack_suffix = if packed { "2" } else { "" };

    let out_ty = match out_opt.elem() {
        Elem::FP8(_) => "e8m0",
        Elem::FP8x2(_) => "e8m0x2",
        _ => unreachable!("Must be scale factor"),
    };

    let in_ty = match input.elem() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::BF16 => format!("bfloat16{pack_suffix}raw"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_NOSAT, cudaRoundPosInf)",
        )
    })
}

fn cast_to_fp8<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
    let out_opt = out.optimized();
    let packing = out.item().vectorization / out_opt.item().vectorization;
    let packed = packing > 1;
    let pack_suffix = if packed { "2" } else { "" };

    let (out_ty, interpretation) = match out_opt.elem() {
        Elem::FP8(kind) => ("fp8", format!("{kind:?}")),
        Elem::FP8x2(kind) => ("fp8x2", format!("{kind:?}")),
        _ => unreachable!("Must be fp8"),
    };

    let in_ty = match input.elem() {
        Elem::F64 => format!("double{pack_suffix}"),
        Elem::TF32 | Elem::F32 => format!("float{pack_suffix}"),
        Elem::BF16 => format!("bfloat16raw{pack_suffix}"),
        Elem::F16 => format!("halfraw{pack_suffix}"),
        _ => unreachable!(),
    };

    let input = input.optimized();

    handle_unroll(f, out, |f, i| {
        let in_value = float_to_packed(input, i, packing);

        write!(
            f,
            "__nv_cvt_{in_ty}_to_{out_ty}({in_value}, __NV_NOSAT, __NV_{interpretation})",
        )
    })
}

fn float_to_packed<D: Dialect>(input: Variable<D>, i: usize, packing: usize) -> String {
    match input.elem() {
        Elem::TF32 | Elem::F32 => {
            let i = i * packing;
            if packing > 1 {
                format!("float2 {{ {}, {} }}", input.index(i), input.index(i + 1))
            } else {
                format!("{}", input.index(i))
            }
        }
        Elem::F64 => {
            let i = i * packing;
            if packing > 1 {
                format!("double2 {{ {}, {} }}", input.index(i), input.index(i + 1))
            } else {
                format!("{}", input.index(i))
            }
        }
        Elem::F16 => format!("{}", input.index(i)),
        Elem::BF16 => format!("{}", input.index(i)),
        _ => unreachable!(),
    }
}

fn cast_minifloat_to_half<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
    let in_opt = input.optimized();
    let out_opt = out.optimized().item();

    let (in_ty, interpretation) = match in_opt.elem() {
        Elem::FP4(kind) => ("fp4", format!("{kind:?}")),
        Elem::FP4x2(kind) => ("fp4x2", format!("{kind:?}")),
        Elem::FP6(kind) => ("fp6", format!("{kind:?}")),
        Elem::FP6x2(kind) => ("fp6x2", format!("{kind:?}")),
        Elem::FP8(kind) => ("fp8", format!("{kind:?}")),
        Elem::FP8x2(kind) => ("fp8x2", format!("{kind:?}")),
        _ => unreachable!("can only cast minifloat"),
    };

    let out_ty = match out_opt.elem() {
        Elem::F16 => "halfraw",
        Elem::F16x2 => "halfraw2",
        _ => unreachable!("out type must be half"),
    };

    handle_unroll(f, out, |f, i| {
        let input = in_opt.index(i);
        write!(
            f,
            "{}(__nv_cvt_{in_ty}_to_{out_ty}({input}, __NV_{interpretation}))",
            out_opt.elem()
        )
    })
}

fn cast_scale_to_bfloat<D: Dialect>(
    f: &mut fmt::Formatter,
    input: Variable<D>,
    out: Variable<D>,
) -> fmt::Result {
    let in_opt = input.optimized();
    let out_opt = out.optimized().item();

    let in_ty = match in_opt.elem() {
        Elem::FP8(_) => "e8m0",
        Elem::FP8x2(_) => "e8m0x2",
        _ => unreachable!("must be scaling factor in e8m0 format"),
    };

    let out_ty = match out_opt.elem() {
        Elem::BF16 => "bf16raw",
        Elem::BF16x2 => "bf162raw",
        _ => unreachable!("out type must be half"),
    };

    handle_unroll(f, out, |f, i| {
        let input = in_opt.index(i);
        write!(
            f,
            "{}(__nv_cvt_{in_ty}_to_{out_ty}({input}))",
            out_opt.elem()
        )
    })
}

fn handle_unroll<D: Dialect>(
    f: &mut fmt::Formatter,
    out: Variable<D>,
    mut op: impl FnMut(&mut fmt::Formatter, usize) -> fmt::Result,
) -> fmt::Result {
    let out_opt = out.item().optimized();
    let vec = out_opt.vectorization;
    let out_var = if out.item() != out_opt {
        Variable::tmp(out_opt)
    } else {
        out
    };
    write!(f, "{} = ", out_var.fmt_left())?;
    if vec > 1 {
        writeln!(f, "{out_opt} {{")?;
    }
    for i in 0..vec {
        op(f, i)?;
        if i + 1 < vec {
            f.write_str(",\n")?;
        }
    }
    if vec > 1 {
        write!(f, "\n}}")?;
    }
    f.write_str(";\n")?;

    if out.item() != out_opt {
        writeln!(
            f,
            "{} = reinterpret_cast<{}&>({out_var});",
            out.fmt_left(),
            out.item()
        )?;
    }
    Ok(())
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
            Elem::F16x2 | Elem::BF16x2 => ("__hmin2", "__hmax2"),
            _ => ("min", "max"),
        };

        let out_fmt = out.fmt_left();
        if num == 1 {
            writeln!(f, "{out_fmt} = ")?;
            D::compile_instruction_max_function_name(f, out.item())?;
            writeln!(f, "({min_value}, ")?;
            D::compile_instruction_min_function_name(f, out.item())?;
            writeln!(f, "({max_value}, {input}));")
        } else {
            writeln!(f, "{out_fmt} = {out_item}{{")?;
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
        let floor = |elem| {
            let prefix = match elem {
                Elem::F16 | Elem::BF16 => D::compile_instruction_half_function_name_prefix(),
                Elem::F16x2 | Elem::BF16x2 => D::compile_instruction_half2_function_name_prefix(),
                _ => "",
            };
            format!("{prefix}floor")
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

            let addr_space = D::address_space_for_variable(&out_tmp);
            let qualifier = out.const_qualifier();
            let out = out.fmt_left();

            writeln!(
                f,
                "{out} = reinterpret_cast<{addr_space}{item_out_original}{qualifier}&>({out_tmp});\n"
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
