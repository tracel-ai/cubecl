use super::{binary::*, unary::*, Component, Elem, Variable, WarpInstruction, WmmaInstruction};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct BinaryInstruction {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone)]
pub struct UnaryInstruction {
    pub input: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Length {
        input: Variable,
        out: Variable,
        num_inputs: usize,
        num_outputs: usize,
    },
    SliceLength {
        input: Variable,
        out: Variable,
    },
    DeclareVariable {
        var: Variable,
    },
    Modulo(BinaryInstruction),
    Remainder(BinaryInstruction),
    Add(BinaryInstruction),
    Fma {
        a: Variable,
        b: Variable,
        c: Variable,
        out: Variable,
    },
    Div(BinaryInstruction),
    Mul(BinaryInstruction),
    Sub(BinaryInstruction),
    Index(BinaryInstruction),
    IndexAssign(BinaryInstruction),
    CheckedIndexAssign(BinaryInstruction),
    Assign(UnaryInstruction),
    RangeLoop {
        i: Variable,
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        instructions: Vec<Self>,
    },
    Loop {
        instructions: Vec<Self>,
    },
    If {
        cond: Variable,
        instructions: Vec<Self>,
    },
    IfElse {
        cond: Variable,
        instructions_if: Vec<Self>,
        instructions_else: Vec<Self>,
    },
    Switch {
        value: Variable,
        instructions_default: Vec<Self>,
        instructions_cases: Vec<(Variable, Vec<Self>)>,
    },
    Slice {
        input: Variable,
        start: Variable,
        end: Variable,
        out: Variable,
    },
    Return,
    Break,
    Stride {
        dim: Variable,
        position: usize,
        out: Variable,
    },
    Shape {
        dim: Variable,
        position: usize,
        out: Variable,
    },
    Equal(BinaryInstruction),
    NotEqual(BinaryInstruction),
    Lower(BinaryInstruction),
    Greater(BinaryInstruction),
    LowerEqual(BinaryInstruction),
    GreaterEqual(BinaryInstruction),
    Erf(UnaryInstruction),
    BitwiseOr(BinaryInstruction),
    BitwiseAnd(BinaryInstruction),
    BitwiseXor(BinaryInstruction),
    ShiftLeft(BinaryInstruction),
    ShiftRight(BinaryInstruction),
    Abs(UnaryInstruction),
    Exp(UnaryInstruction),
    Log(UnaryInstruction),
    Log1p(UnaryInstruction),
    Cos(UnaryInstruction),
    Sin(UnaryInstruction),
    Tanh(UnaryInstruction),
    Powf(BinaryInstruction),
    Sqrt(UnaryInstruction),
    Min(BinaryInstruction),
    Max(BinaryInstruction),
    Not(UnaryInstruction),
    Or(BinaryInstruction),
    And(BinaryInstruction),
    Clamp {
        input: Variable,
        min_value: Variable,
        max_value: Variable,
        out: Variable,
    },
    SyncThreads,
    ThreadFence,
    Round(UnaryInstruction),
    Ceil(UnaryInstruction),
    Floor(UnaryInstruction),
    Wrap(WarpInstruction),
    Wmma(WmmaInstruction),
    Bitcast(UnaryInstruction),
    AtomicLoad(UnaryInstruction),
    AtomicStore(UnaryInstruction),
    AtomicSwap(BinaryInstruction),
    AtomicAdd(BinaryInstruction),
    AtomicSub(BinaryInstruction),
    AtomicMax(BinaryInstruction),
    AtomicMin(BinaryInstruction),
    AtomicAnd(BinaryInstruction),
    AtomicOr(BinaryInstruction),
    AtomicXor(BinaryInstruction),
    AtomicCAS {
        input: Variable,
        cmp: Variable,
        val: Variable,
        out: Variable,
    },
    Negate(UnaryInstruction),
    Magnitude(UnaryInstruction),
    Normalize(UnaryInstruction),
    Dot(BinaryInstruction),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => f.write_str("return;"),
            Instruction::Break => f.write_str("break;"),
            Instruction::DeclareVariable { var } => match var {
                Variable::WmmaFragment {
                    id: _,
                    frag,
                    depth: _,
                } => writeln!(f, "{frag} {var};"),
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
                writeln!(f, "uint {out}_length = {end} - {start};")?;
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
            Instruction::IndexAssign(it) => IndexAssign::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::CheckedIndexAssign(it) => {
                IndexAssign::format(f, &it.lhs, &it.rhs, &it.out)
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
                writeln!(f, "{out} = info[({position} * rank_2) + {dim} + 1];")
            }
            Instruction::Shape { dim, position, out } => writeln!(
                f,
                "{out} = info[({position} * rank_2) + rank + {dim} + 1];"
            ),
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
                match (input.elem(), out.elem()) {
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
            } => writeln!(f, "{out} = atomicCAS({input}, {cmp}, {val});"),
            Instruction::AtomicSwap(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicExch({lhs}, {rhs});")
            }
            Instruction::AtomicAdd(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicAdd({lhs}, {rhs});")
            }
            Instruction::AtomicSub(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicSub({lhs}, {rhs});")
            }
            Instruction::AtomicMax(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicMax({lhs}, {rhs});")
            }
            Instruction::AtomicMin(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicMin({lhs}, {rhs});")
            }
            Instruction::AtomicAnd(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicAnd({lhs}, {rhs});")
            }
            Instruction::AtomicOr(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicOr({lhs}, {rhs});")
            }
            Instruction::AtomicXor(BinaryInstruction { lhs, rhs, out }) => {
                writeln!(f, "{out} = atomicXor({lhs}, {rhs});")
            }
            Instruction::AtomicLoad(UnaryInstruction { input, out }) => {
                writeln!(f, "{out} = atomicAdd({input}, 0);")
            }
            Instruction::AtomicStore(UnaryInstruction { input, out }) => {
                writeln!(f, "atomicExch({out}, {input});")
            }
            Instruction::Remainder(inst) => Remainder::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::Negate(UnaryInstruction { input, out }) => {
                writeln!(f, "{out} = !{input};")
            }
            Instruction::Normalize(inst) => Normalize::format(f, &inst.input, &inst.out),
            Instruction::Magnitude(inst) => Magnitude::format(f, &inst.input, &inst.out),
            Instruction::Dot(inst) => Dot::format(f, &inst.lhs, &inst.rhs, &inst.out),
        }
    }
}

struct Fma;

impl Fma {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        a: &Variable,
        b: &Variable,
        c: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let num = out.item().vectorization;

        for i in 0..num {
            let ai = a.index(i);
            let bi = b.index(i);
            let ci = c.index(i);
            let outi = out.index(i);

            writeln!(f, "{outi} = fma({ai}, {bi}, {ci});")?;
        }

        Ok(())
    }
}

struct Clamp;

impl Clamp {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable,
        min_value: &Variable,
        max_value: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let input = input.optimized();
        let min_value = min_value.optimized();
        let max_value = max_value.optimized();
        let out = out.optimized();
        let num = out.item().vectorization;

        for i in 0..num {
            let inputi = input.index(i);
            let mini = min_value.index(i);
            let maxi = max_value.index(i);
            let outi = out.index(i);

            writeln!(f, "{outi} = max({mini}, min({maxi}, {inputi}));")?;
        }

        Ok(())
    }
}

struct Remainder;

impl Remainder {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let lhs = lhs.optimized();
        let rhs = rhs.optimized();
        let out = out.optimized();
        let num = out.item().vectorization;

        for i in 0..num {
            let lhsi = lhs.index(i);
            let rhsi = rhs.index(i);
            let outi = out.index(i);

            writeln!(f, "{outi} = {lhsi} - {rhsi} * floor({lhsi} / {rhsi});")?;
        }

        Ok(())
    }
}

struct Magnitude;

impl Magnitude {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let num = input.item().vectorization;
        let elem = input.elem();

        writeln!(f, "{out} = 0.0;")?;

        for i in 0..num {
            let input_i = input.index(i);
            writeln!(f, "{out} += {input_i} * {input_i};")?;
        }

        Sqrt::format_unary(f, out, out, elem)
    }
}

struct Normalize;

impl Normalize {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let num = input.item().vectorization;
        let elem = input.elem();
        let norm = format!("{out}_norm");

        writeln!(f, "{{")?;
        writeln!(f, "{elem} {norm} = 0.0;")?;

        for i in 0..num {
            let input_i = input.index(i);
            writeln!(f, "{norm} += {input_i} * {input_i};")?;
        }

        Sqrt::format_unary(f, &norm, &norm, elem)?;

        for i in 0..num {
            let input_i = input.index(i);
            let output_i = out.index(i);

            writeln!(f, "{output_i} = {input_i} / {norm};")?;
        }

        writeln!(f, "}}")
    }
}

struct Dot;

impl Dot {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let num = lhs.item().vectorization;

        writeln!(f, "{out} = 0.0;")?;

        for i in 0..num {
            let lhs_i = lhs.index(i);
            let rhs_i = rhs.index(i);
            writeln!(f, "{out} += {lhs_i} * {rhs_i};")?;
        }
        Ok(())
    }
}
