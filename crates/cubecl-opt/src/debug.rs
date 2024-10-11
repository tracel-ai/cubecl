use std::{collections::HashSet, fmt::Display};

use petgraph::visit::EdgeRef;

use crate::{
    gvn::{BlockNumbers, GlobalNumberGraph},
    passes::{get_out, var_id},
    AtomicCounter, ControlFlow,
};

use super::Optimizer;

/// Debug display for the program state.
impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let post_order = self.post_order.iter().map(|it| format!("bb{}", it.index()));
        let post_order = post_order.collect::<Vec<_>>();
        writeln!(f, "Post Order: {}", post_order.join(", "))?;
        writeln!(f)?;
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

        let mut global_nums = GlobalNumberGraph::default();
        global_nums.available_cfa(self);

        let mut globs = {
            let globals = global_nums.globals.borrow();
            globals.values().copied().collect::<Vec<_>>()
        };
        globs.sort();
        let globs = globs
            .into_iter()
            .map(|it| it.to_string())
            .collect::<Vec<_>>();
        writeln!(f, "Globals: [{}]", globs.join(", "))?;

        for node in self.program.node_indices() {
            let local_nums = global_nums
                .get(&node)
                .cloned()
                .unwrap_or_else(|| BlockNumbers::new(AtomicCounter::default(), Default::default()));
            let avail_in = local_nums.avail_in;
            let mut avail_out = local_nums.avail_out;

            let id = node.index();
            let bb = &self.program[node];
            writeln!(f, "bb{id} {{")?;
            if !bb.block_use.is_empty() {
                writeln!(f, "    Uses: {:?}", bb.block_use)?;
            }
            let live_vars = bb.live_vars.iter();
            let live_vars = live_vars.map(|it| format!("local({}, {})", it.0, it.1));
            let live_vars = live_vars.collect::<Vec<_>>();
            writeln!(f, "    Live variables: [{}]\n", live_vars.join(", "))?;
            let avail = avail_in.classes.values().copied().collect::<HashSet<_>>();
            let mut avail = Vec::from_iter(avail);
            avail.sort();
            let avail = avail
                .into_iter()
                .map(|it| it.to_string())
                .collect::<Vec<_>>();
            writeln!(f, "    Avail classes: [{}]\n", avail.join(", "))?;

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
                let out = get_out(&mut self.clone(), op);
                let id = out.and_then(|var| var_id(&var));
                let range = id.and_then(|id| self.program.int_ranges.get(&id));
                let range = range.map(|it| format!(" range: {it};")).unwrap_or_default();
                let num = avail_out.class_of_operation(op);
                let number = num.map(|it| format!(" class: {it};")).unwrap_or_default();

                writeln!(f, "    {op};{range}{number}")?;
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
