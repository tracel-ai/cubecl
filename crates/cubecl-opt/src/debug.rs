use std::fmt::Display;

use petgraph::visit::EdgeRef;

use crate::ControlFlow;

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
            let live_vars = bb
                .liveness
                .iter()
                .filter(|it| *it.1)
                .map(|it| format!("local({}, {})", it.0 .0, it.0 .1))
                .collect::<Vec<_>>();
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

            for op in bb.ops.borrow().values() {
                writeln!(f, "    {op};")?;
            }
            match &*bb.control_flow.borrow() {
                ControlFlow::Break {
                    cond,
                    body,
                    or_break,
                } => {
                    writeln!(
                        f,
                        "    break {cond} body: bb{}, break: bb{};",
                        body.index(),
                        or_break.index()
                    )?;
                }
                ControlFlow::IfElse {
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
