use std::{fmt::Display, rc::Rc};

use cubecl_ir::{FloatKind, IntKind, UIntKind};
use petgraph::visit::EdgeRef;

use crate::{
    ControlFlow,
    analyses::{liveness::Liveness, uniformity::Uniformity},
    gvn::{BlockSets, Constant, Expression, GlobalValues, Instruction, Local, Value, ValueTable},
};

use super::Optimizer;

const DEBUG_GVN: bool = false;

/// Debug display for the program state.
impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let global_nums = self
            .analysis_cache
            .try_get::<GlobalValues>()
            .unwrap_or_default();
        let liveness = self
            .analysis_cache
            .try_get::<Liveness>()
            .unwrap_or_else(|| Rc::new(Liveness::empty(self)));
        let uniformity = self
            .analysis_cache
            .try_get::<Uniformity>()
            .unwrap_or_default();

        if DEBUG_GVN {
            writeln!(f, "# Value Table:")?;
            writeln!(f, "{}", global_nums.borrow().values)?;
        }

        for node in self.program.node_indices() {
            let id = node.index();
            let bb = &self.program[node];
            let uniform = match uniformity.is_block_uniform(node) {
                true => "uniform ",
                false => "",
            };
            writeln!(f, "{uniform}bb{id} {{")?;
            if DEBUG_GVN {
                let block_sets = &global_nums
                    .borrow()
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
            let live_vars = live_vars.map(|it| format!("local({it})"));
            let live_vars = live_vars.collect::<Vec<_>>();
            writeln!(f, "    Live variables: [{}]\n", live_vars.join(", "))?;

            for phi in bb.phi_nodes.borrow().iter() {
                write!(f, "    {} = phi ", phi.out)?;
                for entry in &phi.entries {
                    write!(f, "[bb{}: ", entry.block.index())?;
                    write!(f, "{}]", entry.value)?;
                }
                let is_uniform = match uniformity.is_var_uniform(phi.out) {
                    true => " @ uniform",
                    false => "",
                };
                writeln!(f, ";{is_uniform}\n")?;
            }
            if !bb.phi_nodes.borrow().is_empty() {
                writeln!(f)?;
            }

            for op in bb.ops.borrow_mut().values_mut() {
                let op_fmt = op.to_string();
                if op_fmt.is_empty() {
                    continue;
                }

                let is_uniform = match op
                    .out
                    .map(|out| uniformity.is_var_uniform(out))
                    .unwrap_or(false)
                {
                    true => " @ uniform",
                    false => "",
                };
                writeln!(f, "    {op_fmt};{is_uniform}")?;
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
        }
    }
}

impl Display for Local {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.version {
            0 => write!(f, "binding({})", self.id),
            v => write!(f, "local({}).v{v}", self.id),
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
        write!(f, "{:?}: [{:?}]", self.op, self.args)
    }
}
