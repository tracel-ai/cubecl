use std::fmt::Display;

use cubecl_core::ir::Variable;
use petgraph::visit::EdgeRef;

use super::{Optimizer, Program};

impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.program)
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "# Variables:")?;
        for ((id, depth), item) in self.variables.iter() {
            let item = match item.vectorization {
                Some(vec) if vec.get() > 1 => {
                    format!("vec{}<{}>", vec.get(), item.elem)
                }
                _ => format!("{}", item.elem),
            };
            writeln!(f, "var local({id}, {depth}): {item:?};")?;
        }
        f.write_str("\n\n")?;

        for node in self.node_indices() {
            let id = node.index();
            let bb = &self[node];
            writeln!(f, "bb{id} {{")?;
            for phi in &bb.phi_nodes {
                write!(
                    f,
                    "    local({}, {}).v{} = ",
                    phi.out.0, phi.out.1, phi.out.2
                )?;
                for entry in &phi.entries {
                    write!(f, "[bb{}: ", entry.block.index())?;
                    let (id, depth, version) = entry.value;
                    write!(f, "local({id}, {depth}).v{version}]  ")?;
                }
                f.write_str(";\n")?;
            }
            for op in &bb.ops {
                writeln!(f, "    {op:?}")?;
            }
            match &bb.control_flow {
                super::ControlFlow::If { cond, then, merge } => {
                    let cond = fmt_var(cond);
                    writeln!(f, "    {cond} ? bb{} : bb{}", then.index(), merge.index())?;
                }
                super::ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    ..
                } => {
                    let cond = fmt_var(cond);
                    writeln!(
                        f,
                        "    {cond} ? bb{} : bb{};",
                        then.index(),
                        or_else.index()
                    )?;
                }
                super::ControlFlow::Switch {
                    value,
                    default,
                    branches,
                    ..
                } => {
                    let value = fmt_var(value);
                    write!(f, "    switch({value}) ")?;
                    for (val, block) in branches {
                        write!(f, "[{val}: bb{}] ", block.index())?;
                    }
                    writeln!(f, "[default: bb{}];", default.index())?;
                }
                super::ControlFlow::Loop { body, .. } => {
                    writeln!(f, "    branch bb{};", body.index())?
                }
                super::ControlFlow::Return => writeln!(f, "    return;")?,
                super::ControlFlow::None => {
                    let edge = self.edges(node).next();
                    let target = edge.map(|it| it.target().index()).unwrap_or(255);
                    writeln!(f, "    branch bb{target};")?;
                }
            }
            f.write_str("}\n\n")?;
        }

        Ok(())
    }
}

fn fmt_var(var: &Variable) -> String {
    match var {
        Variable::GlobalInputArray { id, .. } => format!("input({id})"),
        Variable::GlobalScalar { id, .. } => format!("scalar({id})"),
        Variable::GlobalOutputArray { id, .. } => format!("output({id})"),
        Variable::Local { id, depth, .. } => format!("local({id}, {depth})"),
        Variable::Versioned {
            id, depth, version, ..
        } => format!("local({id}, {depth}).v{version}"),
        Variable::LocalBinding { id, depth, .. } => format!("binding({id}, {depth})"),
        Variable::ConstantScalar(val) => format!("{val:?}"),
        Variable::ConstantArray { id, .. } => format!("const_array({id})"),
        Variable::SharedMemory { id, .. } => format!("shared({id})"),
        Variable::LocalArray { id, .. } => format!("array({id})"),
        Variable::Matrix { id, depth, .. } => format!("matrix({id}, {depth})"),
        Variable::Slice { id, .. } => format!("slice({id})"),
        builtin => format!("{builtin:?}"),
    }
}
