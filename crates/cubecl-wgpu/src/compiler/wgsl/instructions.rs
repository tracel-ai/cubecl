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
    Unreachable,
    WorkgroupBarrier,
    StorageBarrier,
    // Index handles casting to correct local variable.
    Index {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    // Index handles casting to correct local variable.
    Assign {
        input: Variable,
        out: Variable,
    },
    Reference {
        input: Variable,
        out: Variable,
    },
    Load {
        input: Variable,
        out: Variable,
    },
    Store {
        input: Variable,
        out: Variable,
    },
    ModFloor {
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
    Tan {
        input: Variable,
        out: Variable,
    },
    Tanh {
        input: Variable,
        out: Variable,
    },
    Sinh {
        input: Variable,
        out: Variable,
    },
    Cosh {
        input: Variable,
        out: Variable,
    },
    ArcCos {
        input: Variable,
        out: Variable,
    },
    ArcSin {
        input: Variable,
        out: Variable,
    },
    ArcTan {
        input: Variable,
        out: Variable,
    },
    ArcSinh {
        input: Variable,
        out: Variable,
    },
    ArcCosh {
        input: Variable,
        out: Variable,
    },
    ArcTanh {
        input: Variable,
        out: Variable,
    },
    Degrees {
        input: Variable,
        out: Variable,
    },
    Radians {
        input: Variable,
        out: Variable,
    },
    ArcTan2 {
        lhs: Variable,
        rhs: Variable,
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
    InverseSqrt {
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
    TrailingZeros {
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
    Trunc {
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
        ptr: Variable,
        cmp: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicAdd {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicSub {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicMax {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicMin {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicAnd {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicOr {
        ptr: Variable,
        value: Variable,
        out: Variable,
    },
    AtomicXor {
        ptr: Variable,
        value: Variable,
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
    VectorSum {
        input: Variable,
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
    Extract {
        vector: Variable,
        index: Variable,
        out: Variable,
    },
    Insert {
        vector: Variable,
        index: Variable,
        value: Variable,
    },
    CopyBulk {
        source: Variable,
        target: Variable,
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
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} + {rhs};")
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
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = min({lhs}, {rhs});")
            }
            Instruction::Max { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = max({lhs}, {rhs});")
            }
            Instruction::And { lhs, rhs, out } => {
                let vector_size = out.item().vectorization_factor();
                if vector_size > 1 {
                    let item = out.item();
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {item}(")?;
                    for i in 0..vector_size {
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
                let vector_size = out.item().vectorization_factor();
                if vector_size > 1 {
                    let item = out.item();
                    let out = out.fmt_left();
                    writeln!(f, "{out} = {item}(")?;
                    for i in 0..vector_size {
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
            Instruction::Index { lhs, rhs, out } => {
                writeln!(f, "let {out} = &{lhs}[{rhs}];")
            }
            Instruction::CopyBulk {
                source,
                target,
                len,
            } => {
                if *len > 1 {
                    panic!("WGSL doesn't support bulk copy yet");
                }
                writeln!(f, "*{target} = *{source};")
            }
            Instruction::Remainder { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} % {rhs};")
            }
            Instruction::ModFloor { lhs, rhs, out } => {
                let f_type = out.item().with_elem(Elem::F32);
                let ty = out.item();
                let lhs_f = lhs.fmt_cast_to(f_type);
                let rhs_f = rhs.fmt_cast_to(f_type);
                let lhs = lhs.fmt_cast_to(ty);
                let rhs = rhs.fmt_cast_to(ty);
                let out = out.fmt_left();
                let floor = f_type.fmt_cast_to(ty, format!("floor({lhs_f} / {rhs_f})"));
                writeln!(f, "{out} = {lhs} - {rhs} * {floor};")
            }
            Instruction::Sub { lhs, rhs, out } => {
                let lhs = lhs.fmt_cast_to(out.item());
                let rhs = rhs.fmt_cast_to(out.item());
                let out = out.fmt_left();
                writeln!(f, "{out} = {lhs} - {rhs};")
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
            Instruction::InverseSqrt { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = inverseSqrt({input});")
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
            Instruction::Tan { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = tan({input});")
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
            Instruction::Sinh { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = sinh({input});")
            }
            Instruction::Cosh { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = cosh({input});")
            }
            Instruction::ArcCos { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = acos({input});")
            }
            Instruction::ArcSin { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = asin({input});")
            }
            Instruction::ArcTan { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atan({input});")
            }
            Instruction::ArcSinh { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = asinh({input});")
            }
            Instruction::ArcCosh { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = acosh({input});")
            }
            Instruction::ArcTanh { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atanh({input});")
            }
            Instruction::Degrees { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = degrees({input});")
            }
            Instruction::Radians { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = radians({input});")
            }
            Instruction::ArcTan2 { lhs, rhs, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = atan2({lhs}, {rhs});")
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

                if vec_left != vec_right {
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
            Instruction::Reference { input, out } => {
                writeln!(f, "let {out} = &{input};")
            }
            Instruction::Load { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = *{input};")
            }
            Instruction::Store { input, out } => {
                writeln!(f, "*{out} = {input};")
            }
            Instruction::Metadata { info_offset, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = info.static_meta[{info_offset}];")
            }
            Instruction::ExtendedMeta {
                dim,
                info_offset,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(
                    f,
                    "{out} = info.dynamic_meta[info.static_meta[{info_offset}] + {dim}];"
                )
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

                match var.item() {
                    Item::Array(_, length) => {
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
                let out_fmt = out.fmt_left();
                match input.elem() {
                    // 64-bit polyfill: split into upper/lower 32 bits
                    Elem::I64 | Elem::U64 => {
                        let u64_item = input.item().with_elem(Elem::U64);
                        let u32_item = input.item().with_elem(Elem::U32);
                        let input = input.fmt_cast_to(u64_item);
                        writeln!(
                            f,
                            "{out_fmt} = select(countLeadingZeros({u32_item}({input} >> {u32_item}(32u))), 32u + countLeadingZeros({u32_item}({input})), ({input} >> {u32_item}(32u)) == {u64_item}(0));"
                        )
                    }
                    _ => {
                        let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
                        writeln!(f, "{out_fmt} = countLeadingZeros({input});")
                    }
                }
            }
            Instruction::TrailingZeros { input, out } => {
                let out_fmt = out.fmt_left();
                match input.elem() {
                    // 64-bit polyfill: split into upper/lower 32 bits
                    Elem::I64 | Elem::U64 => {
                        let u64_item = input.item().with_elem(Elem::U64);
                        let u32_item = input.item().with_elem(Elem::U32);
                        let input = input.fmt_cast_to(u64_item);
                        writeln!(
                            f,
                            "{out_fmt} = select(countTrailingZeros({u32_item}({input})), 32u + countTrailingZeros({u32_item}({input} >> {u32_item}(32u))), {u32_item}({input}) == {u32_item}(0u));"
                        )
                    }
                    _ => {
                        let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
                        writeln!(f, "{out_fmt} = countTrailingZeros({input});")
                    }
                }
            }
            Instruction::FindFirstSet { input, out } => {
                let out_fmt = out.fmt_left();
                match input.elem() {
                    // 64-bit polyfill: split into upper/lower 32 bits
                    Elem::I64 | Elem::U64 => {
                        let u64_item = input.item().with_elem(Elem::U64);
                        let u32_item = input.item().with_elem(Elem::U32);
                        let input = input.fmt_cast_to(u64_item);
                        writeln!(
                            f,
                            "{out_fmt} = select(firstTrailingBit({u32_item}({input})) + 1, select(firstTrailingBit({u32_item}({input} >> {u32_item}(32u))) + 33, {u32_item}(0u), ({input} >> {u32_item}(32u)) == {u64_item}(0)), {u32_item}({input}) == {u32_item}(0u));"
                        )
                    }
                    _ => {
                        let input = input.fmt_cast_to(input.item().with_elem(Elem::U32));
                        writeln!(f, "{out_fmt} = firstTrailingBit({input}) + 1;")
                    }
                }
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
            Instruction::Trunc { input, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = trunc({input});")
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
            Instruction::AtomicAdd { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicAdd({ptr}, {value});")
            }
            Instruction::AtomicSub { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicSub({ptr}, {value});")
            }
            Instruction::AtomicMax { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicMax({ptr}, {value});")
            }
            Instruction::AtomicMin { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicMin({ptr}, {value});")
            }
            Instruction::AtomicAnd { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicAnd({ptr}, {value});")
            }
            Instruction::AtomicOr { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicOr({ptr}, {value});")
            }
            Instruction::AtomicXor { ptr, value, out } => {
                let out = out.fmt_left();
                write!(f, "{out} = atomicXor({ptr}, {value});")
            }
            Instruction::AtomicCompareExchangeWeak {
                ptr,
                cmp,
                value,
                out,
            } => {
                let out = out.fmt_left();
                writeln!(
                    f,
                    // For compatibility with cuda, only return old_value
                    "{out} = atomicCompareExchangeWeak({ptr}, {cmp}, {value}).old_value;"
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
                    let vec2_type = Item::Vector(out.elem(), 2);
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
            Instruction::VectorSum { input, out } => {
                let vec_size = input.item().vectorization_factor();
                let out = out.fmt_left();
                if vec_size <= 1 {
                    writeln!(f, "{out} = {input};")
                } else {
                    let elems = (0..vec_size)
                        .map(|i| format!("{}", input.index(i)))
                        .collect::<Vec<_>>();
                    writeln!(f, "{out} = {};", elems.join(" + "))
                }
            }
            Instruction::VecInit { inputs, out } => {
                let item = out.item();
                let inputs = inputs.iter().map(|var| var.to_string()).collect::<Vec<_>>();
                let out = out.fmt_left();
                writeln!(f, "{out} = {item}({});", inputs.join(", "))
            }
            Instruction::Extract { vector, index, out } => {
                let out = out.fmt_left();
                writeln!(f, "{out} = {vector}[{index}];")
            }
            Instruction::Insert {
                vector,
                index,
                value,
            } => {
                writeln!(f, "{vector}[{index}] = {value};")
            }
            Instruction::Comment { content } => {
                if content.contains('\n') {
                    writeln!(f, "/* {content} */")
                } else {
                    writeln!(f, "// {content}")
                }
            }
            // WGSL as usual has no lower level intrinsics
            Instruction::Unreachable => writeln!(f, "return;"),
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
