use super::{
    Elem, Subgroup,
    base::{Item, Variable},
};
use std::fmt::Display;

/// All instructions that can be used in a WGSL compute shader.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Instruction {
    DeclareVariable {
        var: Variable,
    },
    Max {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Min {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Add {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Fma {
        a: Variable,
        b: Variable,
        c: Variable,
        out: Variable,
    },
    If {
        cond: Variable,
        instructions: Vec<Instruction>,
    },
    IfElse {
        cond: Variable,
        instructions_if: Vec<Instruction>,
        instructions_else: Vec<Instruction>,
    },
    Select {
        cond: Variable,
        then: Variable,
        or_else: Variable,
        out: Variable,
    },
    Switch {
        value: Variable,
        instructions_default: Vec<Instruction>,
        cases: Vec<(Variable, Vec<Instruction>)>,
    },
    Return,
    Break,
    WorkgroupBarrier,
    StorageBarrier,
    // Index handles casting to correct local variable.
    Index {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    // Index assign handles casting to correct output variable.
    IndexAssign {
        index: Variable,
        rhs: Variable,
        out: Variable,
    },
    // Index handles casting to correct local variable.
    Assign {
        input: Variable,
        out: Variable,
    },
    Modulo {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sub {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Mul {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Div {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Abs {
        input: Variable,
        out: Variable,
    },
    Exp {
        input: Variable,
        out: Variable,
    },
    Log {
        input: Variable,
        out: Variable,
    },
    Log1p {
        input: Variable,
        out: Variable,
    },
    Cos {
        input: Variable,
        out: Variable,
    },
    Sin {
        input: Variable,
        out: Variable,
    },
    Tanh {
        input: Variable,
        out: Variable,
    },
    Powf {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sqrt {
        input: Variable,
        out: Variable,
    },
    Recip {
        input: Variable,
        out: Variable,
    },
    Equal {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Lower {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Clamp {
        input: Variable,
        min_value: Variable,
        max_value: Variable,
        out: Variable,
    },
    Greater {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    LowerEqual {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    GreaterEqual {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    NotEqual {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Length {
        var: Variable,
        out: Variable,
    },
    Metadata {
        info_offset: Variable,
        out: Variable,
    },
    ExtendedMeta {
        info_offset: Variable,
        dim: Variable,
        out: Variable,
    },
    RangeLoop {
        i: Variable,
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        instructions: Vec<Instruction>,
    },
    And {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Or {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Not {
        input: Variable,
        out: Variable,
    },
    Loop {
        instructions: Vec<Instruction>,
    },
    BitwiseOr {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    BitwiseAnd {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    BitwiseXor {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    CountBits {
        input: Variable,
        out: Variable,
    },
    ReverseBits {
        input: Variable,
        out: Variable,
    },
    ShiftLeft {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    ShiftRight {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    BitwiseNot {
        input: Variable,
        out: Variable,
    },
    LeadingZeros {
        input: Variable,
        out: Variable,
    },
    FindFirstSet {
        input: Variable,
        out: Variable,
    },
    Round {
        input: Variable,
        out: Variable,
    },
    Floor {
        input: Variable,
        out: Variable,
    },
    Ceil {
        input: Variable,
        out: Variable,
    },
    Remainder {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Slice {
        input: Variable,
        start: Variable,
        end: Variable,
        out: Variable,
    },
    CheckedSlice {
        input: Variable,
        start: Variable,
        end: Variable,
        out: Variable,
        len: Variable, // The length of the input.
    },
    Bitcast {
        input: Variable,
        out: Variable,
    },
    AtomicLoad {
        input: Variable,
        out: Variable,
    },
    AtomicStore {
        input: Variable,
        out: Variable,
    },
    AtomicSwap {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicCompareExchangeWeak {
        lhs: Variable,
        cmp: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicAdd {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicSub {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicMax {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicMin {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicAnd {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicOr {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AtomicXor {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Subgroup(Subgroup),
    Negate {
        input: Variable,
        out: Variable,
    },
    Magnitude {
        input: Variable,
        out: Variable,
    },
    Normalize {
        input: Variable,
        out: Variable,
    },
    Dot {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    IsNan {
        input: Variable,
        out: Variable,
    },
    IsInf {
        input: Variable,
        out: Variable,
    },
    VecInit {
        inputs: Vec<Variable>,
        out: Variable,
    },
    Copy {
        input: Variable,
        in_index: Variable,
        out: Variable,
        out_index: Variable,
    },
    CopyBulk {
        input: Variable,
        in_index: Variable,
        out: Variable,
        out_index: Variable,
        len: u32,
    },
    Comment {
        content: String,
    },
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::DeclareVariable { var } => {
                let item = var.item();
                writeln!(f, "var {var}: {item};")
            }
            Instruction::Add { lhs, rhs, out } => {
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular addition on atomic");
                    writeln!(f, "atomicAdd({out}, {rhs});")
                } else {
                    let lhs = lhs.fmt_cast_to(out.item());
                    let rhs = rhs.fmt_cast_to(out.item());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {lhs} + {rhs};")
                }
            }
            Instruction::Slice {
                input,
                start,
                end,
                out,
            } => {
                writeln!(f, "let {out}_offset = {start};")?;
                writeln!(f, "let {out}_length = {end} - {start};")?;
                writeln!(f, "let {out}_ptr = &{input};")
            }
            Instruction::CheckedSlice {
                input,
                start,
                end,
                out,
                len,
            } => {
                writeln!(f, "let {out}_offset = {start};")?;
                writeln!(f, "let {out}_length = min({len}, {end}) - {start};")?;
                writeln!(f, "let {out}_ptr = &{input};")
            }
            Instruction::Fma { a, b, c, out } => {
                let a = a.fmt_cast_to(out.item());
                let b = b.fmt_cast_to(out.item());
                let c = c.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = fma({a}, {b}, {c});")
            }
            Instruction::Min { lhs, rhs, out } => {
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular min on atomic");
                    writeln!(f, "atomicMin({out}, {rhs});")
                } else {
                    let lhs = lhs.fmt_cast_to(out.item());
                    let rhs = rhs.fmt_cast_to(out.item());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = min({lhs}, {rhs});")
                }
            }
            Instruction::Max { lhs, rhs, out } => {
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular max on atomic");
                    writeln!(f, "atomicMax({out}, {rhs});")
                } else {
                    let lhs = lhs.fmt_cast_to(out.item());
                    let rhs = rhs.fmt_cast_to(out.item());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = max({lhs}, {rhs});")
                }
            }
            Instruction::And { lhs, rhs, out } => {
                let line_size = out.item().vectorization_factor();
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular and on atomic");
                    writeln!(f, "atomicAnd({out}, {rhs});")
                } else if line_size > 1 {
                    let item = out.item();
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {item}(")?;
                    for i in 0..line_size {
                        let lhs_i = lhs.index(i);
                        let rhs_i = rhs.index(i);
                        writeln!(f, "{lhs_i} && {rhs_i},")?;
                    }
                    writeln!(f, ");")
                } else {
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {lhs} && {rhs};")
                }
            }
            Instruction::Or { lhs, rhs, out } => {
                let line_size = out.item().vectorization_factor();
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular or on atomic");
                    writeln!(f, "atomicOr({out}, {rhs});")
                } else if line_size > 1 {
                    let item = out.item();
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {item}(")?;
                    for i in 0..line_size {
                        let lhs_i = lhs.index(i);
                        let rhs_i = rhs.index(i);
                        writeln!(f, "{lhs_i} || {rhs_i},")?;
                    }
                    writeln!(f, ");")
                } else {
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {lhs} || {rhs};")
                }
            }
            Instruction::Not { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = !{input};")
            }
            Instruction::Index { lhs, rhs, out } => index(f, lhs, rhs, out, None, None),
            Instruction::IndexAssign {
                index: lhs,
                rhs,
                out,
            } => index_assign(f, lhs, rhs, out, None),
            Instruction::Copy {
                input,
                in_index,
                out,
                out_index,
            } => {
                let rhs = format!("{input}[{in_index}]");
                let lhs = format!("{out}[{out_index}]");
                writeln!(f, "{lhs} = {rhs};")
            }
            Instruction::CopyBulk {
                input,
                in_index,
                out,
                out_index,
                len,
            } => {
                for i in 0..*len {
                    let rhs = format!("{input}[{in_index} + {i}]");
                    let lhs = format!("{out}[{out_index} + {i}]");
                    writeln!(f, "{lhs} = {rhs};")?;
                }
                Ok(())
            }
            Instruction::Modulo { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} % {rhs};")
            }
            Instruction::Remainder { lhs, rhs, out } => {
                let f_type = out.item().with_elem(Elem::F32);
                let ty = out.item();
                let lhs = lhs.fmt_cast_to(f_type);
                let rhs = rhs.fmt_cast_to(f_type);
                let out = out.fmt_left();
                let floor = f_type.fmt_cast_to(ty, format!("floor({lhs} / {rhs})"));
                writeln!(f, "{out} = {lhs} - {rhs} * {floor};")
            }
            Instruction::Sub { lhs, rhs, out } => {
                if out.is_atomic() {
                    assert_eq!(lhs, out, "Can't use regular sub on atomic");
                    writeln!(f, "atomicSub({out}, {rhs});")
                } else {
                    let lhs = lhs.fmt_cast_to(out.item());
                    let rhs = rhs.fmt_cast_to(out.item());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {lhs} - {rhs};")
                }
            }
            Instruction::Mul { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} * {rhs};")
            }
            Instruction::Div { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} / {rhs};")
            }
            Instruction::Abs { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = abs({input});")
            }
            Instruction::Exp { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = exp({input});")
            }
            Instruction::Log { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = log({input});")
            }
            Instruction::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => {
                let min = min_value.fmt_cast_to(out.item());
                let max = max_value.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = clamp({input}, {min}, {max});")
            }
            Instruction::Powf { lhs, rhs, out } => super::call_powf(f, lhs, rhs, out),
            Instruction::IsNan { input, out } => super::call_is_nan(f, input, out),
            Instruction::IsInf { input, out } => super::call_is_inf(f, input, out),
            Instruction::Sqrt { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = sqrt({input});")
            }
            Instruction::Log1p { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = log({input} + 1.0);")
            }
            Instruction::Cos { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = cos({input});")
            }
            Instruction::Sin { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = sin({input});")
            }
            Instruction::Tanh { input, out } => {
                #[cfg(target_os = "macos")]
                let result = super::call_safe_tanh(f, input, out);
                #[cfg(not(target_os = "macos"))]
                let result = {
                    let out = out.fmt_left();
                    writeln!(f, "{out} = tanh({input});")
                };

                result
            }
            Instruction::Recip { input, out } => {
                let item = input.item();
                let out = out.fmt_left();
                write!(f, "{out} = {item}(1.0) / {input};")
            }
            Instruction::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            Instruction::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            Instruction::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            Instruction::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            Instruction::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
            Instruction::NotEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "!=", f),
            Instruction::Assign { input, out } => {
                let vec_left = out.item().vectorization_factor();
                let vec_right = input.item().vectorization_factor();

                if out.elem().is_atomic() {
                    if !input.is_atomic() {
                        writeln!(f, "let {out} = {input};")
                    } else {
                        writeln!(f, "let {out} = &{input};")
                    }
                } else if vec_left != vec_right {
                    if vec_right == 1 {
                        let input = input.fmt_cast_to(out.item());
                        let out = out.fmt_left();
                        writeln!(f, "{out} = {input};")
                    } else {
                        for i in 0..vec_right {
                            let out = out.index(i);
                            let input = input.index(i);
                            writeln!(f, "{out} = {input};")?;
                        }
                        Ok(())
                    }
                } else {
                    let input = input.fmt_cast_to(out.item());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {input};")
                }
            }
            Instruction::Metadata { info_offset, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[{info_offset}];")
            }
            Instruction::ExtendedMeta {
                dim,
                info_offset,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info[info[{info_offset}] + {dim}];")
            }
            Instruction::RangeLoop {
                i,
                start,
                end,
                step,
                inclusive,
                instructions,
            } => {
                let increment = step
                    .as_ref()
                    .map(|step| format!("{i} += {step}"))
                    .unwrap_or_else(|| format!("{i}++"));
                let cmp = if *inclusive { "<=" } else { "<" };
                let i_ty = i.item();

                write!(
                    f,
                    "
for (var {i}: {i_ty} = {start}; {i} {cmp} {end}; {increment}) {{
"
                )?;
                for instruction in instructions {
                    write!(f, "{instruction}")?;
                }

                f.write_str("}\n")
            }
            Instruction::If { cond, instructions } => {
                writeln!(f, "if {cond} {{")?;
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
                writeln!(f, "if {cond} {{")?;
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
                let bool_ty = out.item().with_elem(Elem::Bool);

                let cond = cond.fmt_cast_to(bool_ty);
                let then = then.fmt_cast_to(out.item());
                let or_else = or_else.fmt_cast_to(out.item());
                let out = out.fmt_left();

                writeln!(f, "{out} = select({or_else}, {then}, {cond});")
            }
            Instruction::Switch {
                value,
                instructions_default,
                cases,
            } => {
                writeln!(f, "switch({value}) {{")?;
                for (val, block) in cases {
                    writeln!(f, "case {val}: {{")?;
                    for i in block {
                        i.fmt(f)?;
                    }
                    f.write_str("}\n")?;
                }
                f.write_str("default: {\n")?;
                for i in instructions_default {
                    i.fmt(f)?;
                }
                f.write_str("}\n}\n")
            }
            Instruction::Return => f.write_str("return;\n"),
            Instruction::Break => f.write_str("break;\n"),
            Instruction::WorkgroupBarrier => f.write_str("workgroupBarrier();\n"),
            Instruction::StorageBarrier => f.write_str("storageBarrier();\n"),
            Instruction::Length { var, out } => {
                let out = out.fmt_left();

                match var {
                    Variable::ConstantArray(_, _, length) => {
                        writeln!(f, "{out} = {length}u;")
                    }
                    Variable::LocalArray(_, _, length) => {
                        writeln!(f, "{out} = {length}u;")
                    }
                    Variable::SharedMemory(_, _, length) => {
                        writeln!(f, "{out} = {length}u;")
                    }
                    _ => {
                        writeln!(f, "{out} = arrayLength({var});")
                    }
                }
            }
            Instruction::Loop { instructions } => {
                writeln!(f, "loop {{")?;
                for i in instructions {
                    write!(f, "{i}")?;
                }
                f.write_str("}\n")
            }
            Instruction::BitwiseOr { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} | {rhs};")
            }
            Instruction::BitwiseAnd { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} & {rhs};")
            }
            Instruction::BitwiseXor { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} ^ {rhs};")
            }
            Instruction::CountBits { input, out } => {
                let out_item = out.item();
                let out = out.fmt_left();
                match input.elem() == *out_item.elem() {
                    true => writeln!(f, "{out} = countOneBits({input});"),
                    false => writeln!(f, "{out} = {out_item}(countOneBits({input}));"),
                }
            }
            Instruction::ReverseBits { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = reverseBits({input});")
            }
            Instruction::ShiftLeft { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item().with_elem(Elem::U32));
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} << {rhs};")
            }
            Instruction::ShiftRight { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item().with_elem(Elem::U32));
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} >> {rhs};")
            }
            Instruction::BitwiseNot { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = ~{input};")
            }
            Instruction::LeadingZeros { input, out } => {
                let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
                let out = out.fmt_left();
                writeln!(f, "{out} = countLeadingZeros({input});")
            }
            Instruction::FindFirstSet { input, out } => {
                let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
                let out = out.fmt_left();
                writeln!(f, "{out} = firstTrailingBit({input}) + 1;")
            }
            Instruction::Round { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = round({input});")
            }
            Instruction::Floor { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = floor({input});")
            }
            Instruction::Ceil { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = ceil({input});")
            }
            Instruction::Subgroup(op) => write!(f, "{op}"),
            Instruction::Bitcast { input, out } => {
                let elem = out.item();
                let out = out.fmt_left();
                writeln!(f, "{out} = bitcast<{elem}>({input});")
            }
            Instruction::AtomicLoad { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atomicLoad({input});")
            }
            Instruction::AtomicStore { input, out } => {
                writeln!(f, "atomicStore({out},{input});")
            }
            Instruction::AtomicSwap { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicExchange({lhs}, {rhs});")
            }
            Instruction::AtomicAdd { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicAdd({lhs}, {rhs});")
            }
            Instruction::AtomicSub { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicSub({lhs}, {rhs});")
            }
            Instruction::AtomicMax { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicMax({lhs}, {rhs});")
            }
            Instruction::AtomicMin { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicMin({lhs}, {rhs});")
            }
            Instruction::AtomicAnd { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicAnd({lhs}, {rhs});")
            }
            Instruction::AtomicOr { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicOr({lhs}, {rhs});")
            }
            Instruction::AtomicXor { lhs, rhs, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicXor({lhs}, {rhs});")
            }
            Instruction::AtomicCompareExchangeWeak {
                lhs,
                cmp,
                value,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(
                    f,
                    // For compatibility with cuda, only return old_value
                    "{out} = atomicCompareExchangeWeak({lhs}, {cmp}, {value}).old_value;"
                )
            }
            Instruction::Negate { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = -{input};")
            }
            Instruction::Magnitude { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = length({input});")
            }
            Instruction::Normalize { input, out } => {
                if input.item().vectorization_factor() == 1 {
                    // We need a check for vectorization factor 1 here, for compatibility with cuda.
                    // You can almost use sign here, however that does not correctly handle the case for x == 0.0.
                    // Therefore we use normalize with vec2, as there is no way to use a NaN literal in wgsl.
                    let vec2_type = Item::Vec2(out.elem());
                    let out = out.fmt_left();
                    writeln!(f, "{out} = normalize({vec2_type}({input}, 0.0)).x;")
                } else {
                    let out = out.fmt_left();
                    writeln!(f, "{out} = normalize({input});")
                }
            }
            Instruction::Dot { lhs, rhs, out } => {
                let out = out.fmt_left();
                if lhs.item().vectorization_factor() == 1 {
                    writeln!(f, "{out} = {lhs} * {rhs};")
                } else {
                    writeln!(f, "{out} = dot({lhs}, {rhs});")
                }
            }
            Instruction::VecInit { inputs, out } => {
                let item = out.item();
                let inputs = inputs.iter().map(|var| var.to_string()).collect::<Vec<_>>();
                let out = out.fmt_left();
                writeln!(f, "{out} = {item}({});", inputs.join(", "))
            }
            Instruction::Comment { content } => {
                if content.contains('\n') {
                    writeln!(f, "/* {content} */")
                } else {
                    writeln!(f, "// {content}")
                }
            }
        }
    }
}

fn comparison(
    lhs: &Variable,
    rhs: &Variable,
    out: &Variable,
    op: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let item = out.item().with_elem(lhs.elem());
    let lhs = lhs.fmt_cast_to(item);
    let rhs = rhs.fmt_cast_to(item);
    let out = out.fmt_left();
    writeln!(f, "{out} = {lhs} {op} {rhs};")
}

struct IndexOffset {
    var: Variable,
    offset: Option<Variable>,
    index: usize,
}
impl IndexOffset {
    fn new(var: &Variable, offset: &Option<Variable>, index: usize) -> Self {
        Self {
            var: var.clone(),
            offset: offset.clone(),
            index,
        }
    }
}

impl Display for IndexOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = self.var.index(self.index);

        match &self.offset {
            Some(offset) => {
                let offset = offset.index(self.index);
                write!(f, "{var} + {offset}")
            }
            None => write!(f, "{var}"),
        }
    }
}

fn index(
    f: &mut std::fmt::Formatter<'_>,
    lhs: &Variable,
    rhs: &Variable,
    out: &Variable,
    offset: Option<Variable>,
    len: Option<&Variable>,
) -> core::fmt::Result {
    let is_scalar = match lhs {
        Variable::LocalMut { item, .. } => item.vectorization_factor() == 1,
        Variable::LocalConst { item, .. } => item.vectorization_factor() == 1,
        Variable::ConstantScalar(..) => true,
        _ => false,
    };

    let (mut value, index) = if is_scalar {
        (format!("{lhs}"), None)
    } else {
        let value = if let Some(offset) = offset {
            format!("{lhs}[{rhs}+{offset}]")
        } else {
            format!("{lhs}[{rhs}]")
        };

        (value, Some(format!("{rhs}")))
    };

    if out.item().elem().is_atomic() {
        // Atomic values don't support casting or bound checking - we just assign the reference.
        value = format!("&{value}");
        writeln!(f, "let {out} = {value};")
    } else {
        // Check for casting
        if lhs.elem() != out.elem() {
            value = lhs.item().fmt_cast_to(out.item(), value)
        };

        // Check for bounds.
        if let Some(ind) = index
            && let Some(len) = len
        {
            // Note: This is technically not 100% allowed. According to the WebGPU specification,
            // any OOB access is a "dynamic error" which allows "many possible outcomes". In practice,
            // both wgpu and Dawn handle this by either returning dummy data or clamping the index
            // to valid bounds. This means it's harmless to use in a select.
            let out_item = out.item();
            value = format!("select({out_item}(0), {value}, {ind} < {len})");
        };

        let out = out.fmt_left();
        writeln!(f, "{out} = {value};")
    }
}

fn index_assign(
    f: &mut std::fmt::Formatter<'_>,
    lhs: &Variable,
    rhs: &Variable,
    out: &Variable,
    offset: Option<Variable>,
) -> core::fmt::Result {
    match lhs.item() {
        Item::Vec4(elem) => {
            let item = Item::Scalar(elem);
            let lhs0 = IndexOffset::new(lhs, &offset, 0);

            let rhs0 = rhs.index(0).fmt_cast(item);
            let rhs1 = rhs.index(1).fmt_cast(item);
            let rhs2 = rhs.index(2).fmt_cast(item);
            let rhs3 = rhs.index(3).fmt_cast(item);

            write!(f, "{out}[{lhs0}] = vec4({rhs0}, {rhs1}, {rhs2}, {rhs3})")
        }
        Item::Vec3(elem) => {
            let item = Item::Scalar(elem);
            let lhs0 = IndexOffset::new(lhs, &offset, 0);

            let rhs0 = rhs.index(0).fmt_cast(item);
            let rhs1 = rhs.index(1).fmt_cast(item);
            let rhs2 = rhs.index(2).fmt_cast(item);

            writeln!(f, "{out}[{lhs0}] = vec3({rhs0}, {rhs1}, {rhs2});")
        }
        Item::Vec2(elem) => {
            let item = Item::Scalar(elem);
            let lhs0 = IndexOffset::new(lhs, &offset, 0);

            let rhs0 = rhs.index(0).fmt_cast(item);
            let rhs1 = rhs.index(1).fmt_cast(item);

            writeln!(f, "{out}[{lhs0}] = vec2({rhs0}, {rhs1});")
        }
        Item::Scalar(_elem) => {
            let is_array = match out {
                Variable::GlobalInputArray(_, _)
                | Variable::GlobalOutputArray(_, _)
                | Variable::SharedMemory(_, _, _)
                | Variable::LocalArray(_, _, _) => true,
                Variable::Named { is_array, .. } => *is_array,
                _ => false,
            };

            if !is_array {
                let elem_out = out.elem();
                let casting_type = match rhs.item() {
                    Item::Vec4(_) => Item::Vec4(elem_out),
                    Item::Vec3(_) => Item::Vec3(elem_out),
                    Item::Vec2(_) => Item::Vec2(elem_out),
                    Item::Scalar(_) => Item::Scalar(elem_out),
                };
                let rhs = rhs.fmt_cast_to(casting_type);
                if matches!(out.item(), Item::Scalar(_)) {
                    writeln!(f, "{out} = {rhs};")
                } else {
                    writeln!(f, "{out}[{lhs}] = {rhs};")
                }
            } else {
                let item_rhs = rhs.item();
                let item_out = out.item();
                let lhs = IndexOffset::new(lhs, &offset, 0);

                let vectorization_factor = item_out.vectorization_factor();
                if vectorization_factor > item_rhs.vectorization_factor() {
                    let casting_type = Item::Scalar(*item_out.elem());
                    write!(f, "{out}[{lhs}] = vec{vectorization_factor}(")?;
                    for i in 0..vectorization_factor {
                        f.write_str(&rhs.index(i).fmt_cast(casting_type))?;

                        if i < vectorization_factor - 1 {
                            f.write_str(",")?;
                        }
                    }
                    f.write_str(");\n")
                } else {
                    let rhs = rhs.fmt_cast_to(item_out);
                    writeln!(f, "{out}[{lhs}] = {rhs};")
                }
            }
        }
    }
}
