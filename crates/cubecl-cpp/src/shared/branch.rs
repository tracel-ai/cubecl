use core::fmt::Write;
use cubecl_core::ir::{
    ContextExt,
    debug::leaf_line,
    dialect::{branch::*, general::SelectOp},
    prelude::*,
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList, std_deps::path::PathBuf};

use crate::{
    error::EmissionErrors,
    shared::{
        CppValue, OpExtCPP, scoped_block, shared_op, shared_op_with_out, ty::TypeExtCPP,
        unroll::unrolling,
    },
};

/// Marks a kernel with debug data: [`block_to_cpp`] then gives each op the `#line` of its source.
#[derive(Clone, Debug, Default)]
pub struct LineDirectives {
    /// The directory of the relative source files, if one has them on this computer. The
    /// directives join it to each relative path, so a tool finds the file from any directory.
    pub directory: Option<PathBuf>,
}

pub fn block_to_cpp(ctx: &Context, block: Ptr<BasicBlock>) -> String {
    let lines = ctx.try_aux_ty::<LineDirectives>().is_some();
    let mut out = String::new();
    // The file and line that the C++ compiler gives to the next line of `out`, if known.
    let mut next_line: Option<(&PathBuf, u32)> = None;
    let ops = block.deref(ctx).iter(ctx);
    for op in ops {
        // `Display` can't fail, so record the error and let `compile_ir` fail the compilation.
        let cpp = match op.to_cpp(ctx) {
            Ok(cpp) => cpp,
            Err(err) => {
                ctx.aux_ty::<EmissionErrors>().record(err);
                continue;
            }
        };
        if cpp.is_empty() {
            continue;
        }
        if lines {
            next_line = line_directive(ctx, op, next_line, &mut out);
        }
        out.push_str(&cpp);
        next_line = match next_line {
            // A nested block wrote its own directives, so the line after it is not known.
            Some(_) if cpp.contains("#line ") => None,
            // Past `u32::MAX` lines, the next op gets a directive.
            Some((file, line)) => u32::try_from(cpp.matches('\n').count())
                .ok()
                .map(|count| (file, line + count)),
            None => None,
        };
    }
    out
}

/// Writes `#line` for `op` into `out`, unless the next line of `out` already has the source line
/// of `op`. Returns the source line of the next line of `out`. C++ cannot show inlined frames, so
/// the directive has the innermost frame of the location.
fn line_directive<'c>(
    ctx: &'c Context,
    op: Ptr<Operation>,
    next_line: Option<(&'c PathBuf, u32)>,
    out: &mut String,
) -> Option<(&'c PathBuf, u32)> {
    let Some(source) = leaf_line(ctx, &op.deref(ctx).loc()) else {
        return next_line;
    };
    if next_line == Some(source) {
        return next_line;
    }
    // A directive must start a line. The caller can put a block after other text on its line, as
    // `case N: { <block> }`, so an empty `out` also gets the newline.
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let (file, line) = source;
    let _ = write!(out, "#line {line} \"");
    let directory = ctx
        .try_aux_ty::<LineDirectives>()
        .and_then(|lines| lines.directory.as_ref());
    let file = match directory {
        Some(directory) if file.is_relative() => directory.join(file),
        _ => file.clone(),
    };
    push_string_literal(out, &file.display().to_string());
    out.push_str("\"\n");
    Some(source)
}

/// Appends `text` to `out` as the body of a C string literal.
fn push_string_literal(out: &mut String, text: &str) {
    out.extend(text.chars().flat_map(|c| {
        let backslash = matches!(c, '\\' | '"').then_some('\\');
        backslash.into_iter().chain([c])
    }));
}

shared_op!(IfOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let else_block = op.else_block(ctx);
    let mut out = format!("if({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.then_block(ctx)));
    if else_block.deref(ctx).iter(ctx).count() > 1 {
        out.push_str("}\n else {\n");
        out.push_str(&block_to_cpp(ctx, else_block));
    }
    out.push_str("}\n");
    out
});

shared_op!(SwitchOp, |op, ctx| {
    let value = op.value(ctx).name(ctx);
    let mut out = format!("switch({value}) {{\n");
    for (value, block) in op.cases(ctx) {
        let block = block_to_cpp(ctx, block);
        let case = format!("case {}: {{ {block} break; }}\n", value.value().to_i128());
        out.push_str(&case);
    }
    let block = block_to_cpp(ctx, op.default_block(ctx));
    out.push_str(&format!("default: {{ {block} break; }}\n"));
    out.push_str("}\n");
    out
});

// Only relevant for IR structure
shared_op!(YieldOp, |_, _| String::new());
shared_op!(ConditionOp, |op, ctx| {
    format!("return {};", op.condition(ctx).name(ctx))
});

shared_op!(ReturnOp, |op, ctx| {
    if let Some(value) = op.value(ctx) {
        format!("return {};", value.name(ctx))
    } else {
        "return;".into()
    }
});

shared_op!(UnreachableOp, |_, _| "__builtin_unreachable();".into());

shared_op!(RangeLoopOp, |op, ctx| {
    let i = op.iter_var(ctx).name(ctx);
    let i_ty = op.iter_var(ctx).get_type(ctx).to_cpp(ctx);
    let start = op.start(ctx).name(ctx);
    let end = op.end(ctx).name(ctx);
    let step = op.step(ctx).name(ctx);
    let mut out = format!("for({i_ty} {i} = {start}; {i} < {end}; {i} += {step}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op!(WhileOp, |op, ctx| {
    let cond = scoped_block! {
        block_to_cpp(ctx, op.before_block(ctx))
    };
    let mut out = format!("while({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.after_block(ctx)));
    out.push_str("}\n");
    out
});

shared_op_with_out!(SelectOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let then = op.true_value(ctx).name(ctx);
    let or_else = op.false_value(ctx).name(ctx);
    format!("{} ? {} : {}", cond, then, or_else)
});
unrolling!(SelectOp);

#[cfg(test)]
mod tests {
    use super::push_string_literal;
    use crate::{
        shared::{CompilationOptions, CppCompiler, register_supported_types},
        target::Cuda,
    };
    use cubecl_core::{
        Compiler,
        ir::settings::DebugInfo,
        runtime_tests::offline::{
            SOURCE_PATH, nested_calls_kernel, offline_device_properties, source_line,
        },
    };
    use cubecl_runtime::kernel::CubeKernel;
    use std::sync::Arc;

    /// The CUDA source of `nested_calls` at the debug level `level`. The level is set after the
    /// resolution, which gives a `dev` build at least line tables, as `CUBECL_DEBUG_INFO` does.
    fn source(level: DebugInfo) -> String {
        let mut properties = offline_device_properties(32);
        register_supported_types(&mut properties);
        let mut definition = nested_calls_kernel(Arc::new(properties), level).define();
        definition.settings.debug_info = level;
        CppCompiler::<Cuda>::default()
            .compile(definition, &CompilationOptions::default())
            .unwrap()
            .to_string()
    }

    /// A file and a line in it.
    type SourceLine<'a> = (&'a str, u32);

    /// Whether a code line is a statement, and the text of the statement in the source.
    type Statement = (fn(&str) -> bool, &'static str);

    /// Each code line of `source`, with the file and the line that the C++ compiler gives it after
    /// the `#line` directives.
    fn compiled_lines(source: &str) -> Vec<(&str, Option<SourceLine<'_>>)> {
        let mut next = None;
        source
            .lines()
            .filter_map(|line| {
                if let Some(rest) = line.strip_prefix("#line ") {
                    let (number, file) = rest.split_once(' ').unwrap();
                    next = Some((file.trim_matches('"'), number.parse().unwrap()));
                    return None;
                }
                let current = next;
                next = next.map(|(file, number)| (file, number + 1));
                Some((line, current))
            })
            .collect()
    }

    /// Each statement gets the line of its innermost `#[cube]` frame, also where a directive is
    /// left out because the line count already gives that line, and after a call returns.
    #[test]
    fn statements_have_the_lines_of_their_source() {
        let source = source(DebugInfo::LineTables);
        let starts_a_line = source
            .match_indices("#line")
            .all(|(at, _)| at == 0 || source.as_bytes()[at - 1] == b'\n');
        assert!(
            starts_a_line,
            "a directive is not at the start of a line:\n{source}"
        );

        let lines = compiled_lines(&source);
        let statements: [Statement; 3] = [
            (
                |code| code.contains("= x_v") && code.contains(" * x_v"),
                "let y = x * x;",
            ),
            (
                |code| code.contains("= y_v") && code.contains(" / "),
                "y / 3.0",
            ),
            (
                |code| code.contains("= third_v") && code.contains(" * "),
                "third * 2.0",
            ),
        ];
        for (is_statement, text) in statements {
            let [(code, line)] = lines
                .iter()
                .filter(|(code, _)| is_statement(code))
                .collect::<Vec<_>>()[..]
            else {
                panic!("no single statement for `{text}` in:\n{source}");
            };
            let (file, number) = line.unwrap_or_else(|| panic!("`{code}` has no line"));
            assert_eq!(number, source_line(text), "`{code}` in:\n{source}");
            assert!(file.ends_with(SOURCE_PATH), "{file}");
            // With `std`, the search finds the workspace root, a parent of the working directory.
            if cfg!(feature = "std") {
                let path = std::path::Path::new(file);
                assert!(path.is_absolute() && path.is_file(), "{file}");
            }
        }
    }

    /// Without debug data, the source has no directive, although the ops have locations.
    #[test]
    fn no_line_directives_without_debug_data() {
        let source = source(DebugInfo::None);
        assert!(!source.contains("#line"), "{source}");
    }

    #[test]
    fn file_names_are_c_string_literals() {
        let mut out = String::new();
        push_string_literal(&mut out, r#"C:\src\"k".rs"#);
        assert_eq!(out, r#"C:\\src\\\"k\".rs"#);
    }
}
