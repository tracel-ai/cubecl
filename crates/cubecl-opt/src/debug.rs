use std::fmt::Display;

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
            writeln!(f, "var local({id}, {depth}): {item};")?;
        }
        f.write_str("\n\n")?;

        for node in self.node_indices() {
            let id = node.index();
            let bb = &self[node];
            writeln!(f, "bb{id} {{")?;
            for phi in &bb.phi_nodes {
                write!(
                    f,
                    "    local({}, {}).v{} = phi ",
                    phi.out.0, phi.out.1, phi.out.2
                )?;
                for entry in &phi.entries {
                    write!(f, "[bb{}: ", entry.block.index())?;
                    let (id, depth, version) = entry.value;
                    write!(f, "local({id}, {depth}).v{version}]")?;
                }
                f.write_str(";\n")?;
            }
            for op in bb.ops.borrow().values() {
                writeln!(f, "    {op};")?;
            }
            match &bb.control_flow {
                super::ControlFlow::If { cond, then, merge } => {
                    writeln!(f, "    {cond} ? bb{} : bb{};", then.index(), merge.index())?;
                }
                super::ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    merge,
                } => {
                    writeln!(
                        f,
                        "    {cond} ? bb{} : bb{}; merge: bb{}",
                        then.index(),
                        or_else.index(),
                        merge.index()
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
