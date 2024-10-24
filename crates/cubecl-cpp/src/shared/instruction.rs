use crate::shared::FmtLeft;

use super::{
    binary::*, unary::*, Component, Dialect, Elem, Variable, WarpInstruction, WmmaInstruction,
};
use std::{fmt::Display, marker::PhantomData};

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
    Length {
        input: Variable<D>,
        out: Variable<D>,
        num_inputs: usize,
        num_outputs: usize,
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
    Return,
    Break,
    Stride {
        dim: Variable<D>,
        position: usize,
        out: Variable<D>,
    },
    Shape {
        dim: Variable<D>,
        position: usize,
        out: Variable<D>,
    },
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
    ShiftLeft(BinaryInstruction<D>),
    ShiftRight(BinaryInstruction<D>),
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
    Round(UnaryInstruction<D>),
    Ceil(UnaryInstruction<D>),
    Floor(UnaryInstruction<D>),
    Wrap(WarpInstruction<D>),
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
    Negate(UnaryInstruction<D>),
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
            Instruction::Mul(it) => Mul::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Div(it) => Div::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sub(it) => Sub::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Modulo(inst) => Modulo::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::BitwiseOr(it) => BitwiseOr::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::BitwiseAnd(it) => BitwiseAnd::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::BitwiseXor(it) => BitwiseXor::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::ShiftLeft(it) => ShiftLeft::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::ShiftRight(it) => ShiftRight::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Index(it) => Index::format(f, &it.lhs, &it.rhs, &it.out),
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
            Instruction::IndexAssign(it) => IndexAssign::format(f, &it.lhs, &it.rhs, &it.out),
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
                let vf_then = then.item().vectorization;
                let vf_or_else = or_else.item().vectorization;
                let vf_out = out.item().vectorization;
                let vf_cond = cond.item().vectorization;

                let vf = usize::max(vf_cond, vf_out);
                let vf = usize::max(vf, vf_then);
                let vf = usize::max(vf, vf_or_else);

                let item_out = out.item();
                let cond_elem = cond.item().elem;
                let out = out.fmt_left();

                if vf > 1 {
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
            Instruction::Stride { dim, position, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[({position} * rank_2) + {dim} + 1];")
            }
            Instruction::Shape { dim, position, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[({position} * rank_2) + rank + {dim} + 1];")
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
            Instruction::Length {
                input,
                out,
                num_inputs,
                num_outputs,
            } => {
                let offset = num_inputs + num_outputs;
                let index = match input {
                    Variable::GlobalInputArray(index, _) => *index as usize,
                    Variable::GlobalOutputArray(index, _) => *index as usize + num_inputs,
                    _ => panic!("Can only know the len of a global array."),
                } + 1;
                let factor = input.item().vectorization;
                let out = out.fmt_left();

                if factor == 1 {
                    return writeln!(f, "{out} = info[({offset} * 2 * info[0]) + {index}];");
                }

                writeln!(
                    f,
                    "{out} = info[({offset} * 2 * info[0]) + {index}] / {factor};"
                )
            }
            Instruction::Wrap(it) => write!(f, "{it}"),
            Instruction::Fma { a, b, c, out } => Fma::format(f, a, b, c, out),
            Instruction::Wmma(it) => write!(f, "{it}"),
            Instruction::Bitcast(UnaryInstruction { input, out }) => {
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
                    _ => panic!("Unsupported type for bitcasting"),
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
                writeln!(f, "{out} = atomicAdd({lhs}, {rhs});")
            }
            Instruction::AtomicSub(BinaryInstruction { lhs, rhs, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicSub({lhs}, {rhs});")
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
                let out = out.fmt_left();
                writeln!(f, "atomicExch({out}, {input});")
            }
            Instruction::Remainder(inst) => Remainder::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::Negate(UnaryInstruction { input, out }) => {
                let out = out.fmt_left();
                writeln!(f, "{out} = !{input};")
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
        }
    }
}

struct Fma<D: Dialect> {
    dialect: PhantomData<D>,
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
    dialect: PhantomData<D>,
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

        let out = out.fmt_left();
        if num == 1 {
            writeln!(f, "{out} = max({min_value}, min({max_value}, {input}));")
        } else {
            writeln!(f, "{out} = {out_item}{{")?;
            for i in 0..num {
                let inputi = input.index(i);
                let mini = min_value.index(i);
                let maxi = max_value.index(i);

                writeln!(f, "max({mini}, min({maxi}, {inputi})),")?;
            }

            f.write_str("};\n")
        }
    }
}

struct Remainder<D: Dialect> {
    dialect: PhantomData<D>,
}

impl<D: Dialect> Remainder<D> {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> core::fmt::Result {
        let lhs = lhs.optimized();
        let rhs = rhs.optimized();
        let out = out.optimized();
        let out_item = out.item();
        let num = out_item.vectorization;

        let out = out.fmt_left();
        if num == 1 {
            writeln!(f, "{out} = {lhs} - {rhs} * floor({lhs} / {rhs});")
        } else {
            writeln!(f, "{out} = {out_item}{{")?;
            for i in 0..num {
                let lhsi = lhs.index(i);
                let rhsi = rhs.index(i);

                writeln!(f, "{lhsi} - {rhsi} * floor({lhsi} / {rhsi}),")?;
            }
            f.write_str("};\n")
        }
    }
}

struct Magnitude<D: Dialect> {
    dialect: PhantomData<D>,
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

        write!(f, "{out} = ")?;
        Sqrt::format_unary(f, &mag, elem)?;
        f.write_str(";\n")
    }
}

struct Normalize<D: Dialect> {
    dialect: PhantomData<D>,
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
    dialect: PhantomData<D>,
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

impl<'a, V: Display, D: Dialect> Display for EnsureBoolArg<'a, V, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elem != &Elem::Bool {
            write!(f, "bool({})", self.var)
        } else {
            write!(f, "{}", self.var)
        }
    }
}
