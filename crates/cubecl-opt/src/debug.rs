use std::{fmt::Display, rc::Rc};

use cubecl_core::ir::{FloatKind, IntKind, UIntKind};
use petgraph::visit::EdgeRef;

use crate::{
    analyses::liveness::Liveness,
    gvn::{BlockSets, Constant, Expression, GvnState, Instruction, Local, OpId, Value, ValueTable},
    passes::var_id,
    ControlFlow,
};

use super::Optimizer;

const DEBUG_GVN: bool = false;

/// Debug display for the program state.
impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Slices:\n")?;
        for (var_id, slice) in self.program.slices.iter() {
            let end_op = slice.end_op.as_ref().map(|it| format!("{it}"));
            writeln!(
                f,
                "slice{var_id:?}: {{ start: {}, end: {}, end_op: {}, const_len: {:?} }}",
                slice.start,
                slice.end,
                end_op.unwrap_or("None".to_string()),
                slice.const_len
            )?;
        }
        f.write_str("\n\n")?;

        let global_nums = self.analyses.try_get::<GvnState>().unwrap_or_default();
        let liveness = self
            .analyses
            .try_get::<Liveness>()
            .unwrap_or_else(|| Rc::new(Liveness::empty(self)));

        if DEBUG_GVN {
            writeln!(f, "# Value Table:")?;
            writeln!(f, "{}", global_nums.values)?;
        }

        for node in self.program.node_indices() {
            let id = node.index();
            let bb = &self.program[node];
            writeln!(f, "bb{id} {{")?;
            if DEBUG_GVN {
                let block_sets = &global_nums
                    .block_sets
                    .get(&node)
                    .cloned()
                    .unwrap_or_default();
                writeln!(f, "{block_sets}")?;
            }

            if !bb.block_use.is_empty() {
                writeln!(f, "    Uses: {:?}", bb.block_use)?;
            }
            let live_vars = liveness.at_block(node).iter();
            let live_vars = live_vars.map(|it| format!("local({}, {})", it.0, it.1));
            let live_vars = live_vars.collect::<Vec<_>>();
            writeln!(f, "    Live variables: [{}]\n", live_vars.join(", "))?;

            for phi in bb.phi_nodes.borrow().iter() {
                write!(f, "    {} = phi ", phi.out)?;
                for entry in &phi.entries {
                    write!(f, "[bb{}: ", entry.block.index())?;
                    write!(f, "{}]", entry.value)?;
                }
                f.write_str(";\n")?;
            }
            if !bb.phi_nodes.borrow().is_empty() {
                writeln!(f)?;
            }

            for op in bb.ops.borrow_mut().values_mut() {
                let id = op.out.and_then(|var| var_id(&var));
                let range = id.and_then(|id| self.program.int_ranges.get(&id));
                let range = range.map(|it| format!(" range: {it};")).unwrap_or_default();

                writeln!(f, "    {op};{range}")?;
            }
            match &*bb.control_flow.borrow() {
                ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    merge,
                } => {
                    writeln!(
                        f,
                        "    {cond} ? bb{} : bb{}; merge: {}",
                        then.index(),
                        or_else.index(),
                        merge
                            .as_ref()
                            .map(|it| format!("bb{}", it.index()))
                            .unwrap_or("None".to_string())
                    )?;
                }
                super::ControlFlow::Switch {
                    value,
                    default,
                    branches,
                    ..
                } => {
                    write!(f, "    switch({value}) ")?;
                    for (val, block) in branches {
                        write!(f, "[{val}: bb{}] ", block.index())?;
                    }
                    writeln!(f, "[default: bb{}];", default.index())?;
                }
                super::ControlFlow::Loop {
                    body,
                    continue_target,
                    merge,
                } => {
                    writeln!(
                        f,
                        "    loop(continue: bb{}, merge: bb{})",
                        continue_target.index(),
                        merge.index()
                    )?;
                    writeln!(f, "    branch bb{};", body.index())?
                }
                super::ControlFlow::LoopBreak {
                    break_cond,
                    body,
                    continue_target,
                    merge,
                } => {
                    writeln!(
                        f,
                        "    loop(cond: {}, body: bb{} continue: bb{}, break: bb{})",
                        break_cond,
                        body.index(),
                        continue_target.index(),
                        merge.index()
                    )?;
                }
                super::ControlFlow::Return => writeln!(f, "    return;")?,
                super::ControlFlow::None => {
                    let edge = self.program.edges(node).next();
                    let target = edge.map(|it| it.target().index()).unwrap_or(255);
                    writeln!(f, "    branch bb{target};")?;
                }
            }
            f.write_str("}\n\n")?;
        }

        Ok(())
    }
}

impl Display for BlockSets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut exp_gen = self.exp_gen.iter().collect::<Vec<_>>();
        exp_gen.sort_by_key(|it| it.0);
        let exp_gen = exp_gen
            .into_iter()
            .map(|(val, expr)| format!("{val}: {expr}"))
            .collect::<Vec<_>>();
        let mut phi_gen = self.phi_gen.iter().collect::<Vec<_>>();
        phi_gen.sort_by_key(|it| it.0);
        let phi_gen = phi_gen
            .into_iter()
            .map(|(val, expr)| format!("{val}: {expr}"))
            .collect::<Vec<_>>();
        let tmp_gen = self
            .tmp_gen
            .iter()
            .map(|it| format!("{it}"))
            .collect::<Vec<_>>();
        let mut leaders = self.leaders.iter().collect::<Vec<_>>();
        leaders.sort_by_key(|it| it.0);
        let leaders = leaders
            .into_iter()
            .map(|(val, expr)| format!("{val}: {expr}"))
            .collect::<Vec<_>>();
        let mut antic_out = self.antic_out.iter().collect::<Vec<_>>();
        antic_out.sort_by_key(|it| it.0);
        let antic_out = antic_out
            .into_iter()
            .map(|(val, expr)| format!("{val}: {expr}"))
            .collect::<Vec<_>>();
        let mut antic_in = self.antic_in.iter().collect::<Vec<_>>();
        antic_in.sort_by_key(|it| it.0);
        let antic_in = antic_in
            .into_iter()
            .map(|(val, expr)| format!("{val}: {expr}"))
            .collect::<Vec<_>>();

        writeln!(f, "    exp_gen: [{}]", exp_gen.join(", "))?;
        writeln!(f, "    phi_gen: [{}]", phi_gen.join(", "))?;
        writeln!(f, "    tmp_gen: [{}]", tmp_gen.join(", "))?;
        writeln!(f, "    leaders: [{}]", leaders.join(", "))?;
        writeln!(f, "    antic_in: [{}]", antic_in.join(", "))?;
        writeln!(f, "    antic_out: [{}]", antic_out.join(", "))
    }
}

impl Display for ValueTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut values = self.value_numbers.iter().collect::<Vec<_>>();
        values.sort_by_key(|it| it.1);
        writeln!(f, "values: [")?;
        for (val, num) in values {
            writeln!(f, "    {num}: {val},")?;
        }
        writeln!(f, "]")?;
        writeln!(f, "expressions: [")?;
        let mut expressions = self.expression_numbers.iter().collect::<Vec<_>>();
        expressions.sort_by_key(|it| it.1);
        for (expr, val) in expressions {
            writeln!(f, "    {val}: {expr},")?;
        }
        writeln!(f, "]")
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Constant(constant) => write!(f, "{constant}"),
            Value::Local(local) => write!(f, "{local}"),
            Value::Input(id, _) => write!(f, "input({id})"),
            Value::Scalar(id, elem) => write!(f, "scalar({elem}, {id})"),
            Value::ConstArray(id, _, _) => write!(f, "const_array({id})"),
            Value::Builtin(builtin) => write!(f, "{builtin:?}"),
            Value::Output(id, _) => write!(f, "output({id})"),
            Value::Slice(id, depth, _) => write!(f, "slice({id}, {depth})"),
        }
    }
}

impl Display for Local {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.version {
            0 => write!(f, "binding({}, {})", self.id, self.depth),
            v => write!(f, "local({}, {}).v{v}", self.id, self.depth),
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Int(val, IntKind::I8) => write!(f, "{val}i8"),
            Constant::Int(val, IntKind::I16) => write!(f, "{val}i16"),
            Constant::Int(val, IntKind::I32) => write!(f, "{val}i32"),
            Constant::Int(val, IntKind::I64) => write!(f, "{val}i64"),
            Constant::Float(val, FloatKind::BF16) => write!(f, "{}bf16", val.0),
            Constant::Float(val, FloatKind::F16) => write!(f, "{}f16", val.0),
            Constant::Float(val, FloatKind::Flex32) => write!(f, "{}minf16", val.0),
            Constant::Float(val, FloatKind::TF32) => write!(f, "{}tf32", val.0),
            Constant::Float(val, FloatKind::F32) => write!(f, "{}f32", val.0),
            Constant::Float(val, FloatKind::F64) => write!(f, "{}f64", val.0),
            Constant::UInt(val, UIntKind::U8) => write!(f, "{val}u8"),
            Constant::UInt(val, UIntKind::U16) => write!(f, "{val}u16"),
            Constant::UInt(val, UIntKind::U32) => write!(f, "{val}u32"),
            Constant::UInt(val, UIntKind::U64) => write!(f, "{val}u64"),
            Constant::Bool(val) => write!(f, "{val}"),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Instruction(instruction) => write!(f, "{instruction}"),
            Expression::Copy(val, _) => write!(f, "copy({val})"),
            Expression::Value(value) => write!(f, "{value}"),
            Expression::Volatile(value) => write!(f, "volatile({value})"),
            Expression::Phi(entries) => write!(
                f,
                "phi({})",
                entries
                    .iter()
                    .map(|(val, b)| format!("{val}: bb{}", b.index()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args = &self.args;
        match self.op {
            OpId::Add => write!(f, "{} + {}", args[0], args[1]),
            OpId::Fma => write!(f, "fma({}, {}, {})", args[0], args[1], args[2]),
            OpId::Sub => write!(f, "{} - {}", args[0], args[1]),
            OpId::Mul => write!(f, "{} * {}", args[0], args[1]),
            OpId::Div => write!(f, "{} / {}", args[0], args[1]),
            OpId::Abs => write!(f, "{}.abs()", args[0]),
            OpId::Exp => write!(f, "{}.exp()", args[0]),
            OpId::Log => write!(f, "{}.log()", args[0]),
            OpId::Log1p => write!(f, "{}.log1p()", args[0]),
            OpId::Cos => write!(f, "{}.cos()", args[0]),
            OpId::Sin => write!(f, "{}.sin()", args[0]),
            OpId::Tanh => write!(f, "{}.tanh()", args[0]),
            OpId::Powf => write!(f, "{}.powf()", args[0]),
            OpId::Sqrt => write!(f, "{}.sqrt()", args[0]),
            OpId::Round => write!(f, "{}.round()", args[0]),
            OpId::Floor => write!(f, "{}.floor()", args[0]),
            OpId::Ceil => write!(f, "{}.ceil()", args[0]),
            OpId::Erf => write!(f, "{}.erf()", args[0]),
            OpId::Recip => write!(f, "1.0 / {}", args[0]),
            OpId::Equal => write!(f, "{} == {}", args[0], args[1]),
            OpId::NotEqual => write!(f, "{} != {}", args[0], args[1]),
            OpId::Lower => write!(f, "{} < {}", args[0], args[1]),
            OpId::Clamp => write!(f, "clamp({}, {}, {})", args[0], args[1], args[2]),
            OpId::Greater => write!(f, "{} > {}", args[0], args[1]),
            OpId::LowerEqual => write!(f, "{} <= {}", args[0], args[1]),
            OpId::GreaterEqual => write!(f, "{} >= {}", args[0], args[1]),
            OpId::Modulo => write!(f, "{} % {}", args[0], args[1]),
            OpId::Index => write!(f, "{}[{}]", args[0], args[1]),
            OpId::InitLine => write!(
                f,
                "vec{}({})",
                args.len(),
                args.iter()
                    .map(|it| it.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            OpId::And => write!(f, "{} && {}", args[0], args[1]),
            OpId::Or => write!(f, "{} || {}", args[0], args[1]),
            OpId::Not => write!(f, "!{}", args[0]),
            OpId::Neg => write!(f, "-{}", args[0]),
            OpId::Max => write!(f, "max({}, {})", args[0], args[1]),
            OpId::Min => write!(f, "min({}, {})", args[0], args[1]),
            OpId::BitwiseAnd => write!(f, "{} & {}", args[0], args[1]),
            OpId::BitwiseOr => write!(f, "{} | {}", args[0], args[1]),
            OpId::BitwiseXor => write!(f, "{} ^ {}", args[0], args[1]),
            OpId::ShiftLeft => write!(f, "{} << {}", args[0], args[1]),
            OpId::ShiftRight => write!(f, "{} >> {}", args[0], args[1]),
            OpId::Remainder => write!(f, "{} % {}", args[0], args[1]),
            OpId::Magnitude => write!(f, "{}.length()", args[0]),
            OpId::Normalize => write!(f, "{}.normalize()", args[0]),
            OpId::Dot => write!(f, "dot({}, {})", args[0], args[1]),
            OpId::Select => write!(f, "select({}, {}, {})", args[0], args[1], args[2]),
            OpId::Bitcast => write!(f, "bitcast<{}>({})", self.item, args[0]),
            OpId::Rank => write!(f, "{}.rank()", args[0]),
            OpId::Length => write!(f, "{}.len()", args[0]),
            OpId::BufferLength => write!(f, "buffer_len({})", args[0]),
            OpId::Shape => write!(f, "{}.shape[{}]", args[0], args[1]),
            OpId::Stride => write!(f, "{}.stride[{}]", args[0], args[1]),
            OpId::Cast => write!(f, "cast<{}>({})", self.item, args[0]),
        }
    }
}
