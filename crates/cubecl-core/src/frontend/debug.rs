use alloc::{string::String, vec::Vec};
use cubecl_ir::{
    dialect::general::PrintfOp,
    pliron::{location::Source, value::Value},
};

use crate::ir::Scope;

use super::CubeDebug;

// The macro calls these functions with plain values: a guard or a closure for each expression
// adds unwind paths and instances to every expand function, which multiplies the compile memory
// of crates with many generic `#[cube]` functions.

/// Moves the current `#[cube]` function to `line` and `column`. The macro calls it after the
/// operands of an expression, before the expression emits its op, so no position needs to be
/// restored.
pub fn debug_pos_expand(scope: &Scope, line: u32, column: u32) {
    if let Some(debug) = scope.debug_state() {
        debug.set_pos(line, column);
    }
}

/// Calls a function from `line` and `column` of the current `#[cube]` function.
pub fn debug_call_expand<C>(
    scope: &Scope,
    line: u32,
    column: u32,
    call: impl FnOnce(&Scope) -> C,
) -> C {
    debug_pos_expand(scope, line, column);
    call(scope)
}

/// Opens the frame of the `#[cube]` function `name`, defined at `line` and `column` of `file`.
/// The frame closes when the returned guard drops.
pub fn debug_source_expand<'a>(
    scope: &'a Scope,
    name: &'static str,
    file: &'static str,
    source_text: &'static str,
    line: u32,
    column: u32,
) -> DebugFrame<'a> {
    let Some(debug) = scope.debug_state() else {
        return DebugFrame { scope: None };
    };
    let path = file.replace('\\', "/");
    debug.add_source(&path, source_text);
    let file = Source::new_from_file(scope.ctx_mut(), path);
    if let Some(debug) = scope.debug_state() {
        debug.enter_fn(name, file, line, column);
    }
    DebugFrame { scope: Some(scope) }
}

/// Closes the frame that [`debug_source_expand`] opened.
pub struct DebugFrame<'a> {
    scope: Option<&'a Scope>,
}

impl Drop for DebugFrame<'_> {
    fn drop(&mut self) {
        if let Some(debug) = self.scope.and_then(Scope::debug_state) {
            debug.exit_fn();
        }
    }
}

/// A variable that the macro names only when its type implements [`CubeDebug`].
///
/// A `#[cube]` function can bind a value of a plain Rust type, for example a comptime enum that
/// another `#[cube]` function returns. That value has no name to set, and must not need
/// [`CubeDebug`]. The macro calls `debug_var` on `&DebugVar`: method resolution takes
/// [`DebugVarNamed`] when it applies, else [`DebugVarUnnamed`].
#[doc(hidden)]
pub struct DebugVar<'a, E>(pub &'a E);

/// Names a [`DebugVar`] whose type implements [`CubeDebug`].
#[doc(hidden)]
pub trait DebugVarNamed {
    fn debug_var(&self, scope: &Scope, name: &'static str);
}

impl<E: CubeDebug> DebugVarNamed for DebugVar<'_, E> {
    fn debug_var(&self, scope: &Scope, name: &'static str) {
        if scope.debug_state().is_some() {
            self.0.set_debug_name(scope, name);
        }
    }
}

/// Leaves a [`DebugVar`] of any other type without a name.
#[doc(hidden)]
pub trait DebugVarUnnamed {
    fn debug_var(&self, _scope: &Scope, _name: &'static str) {}
}

impl<E> DebugVarUnnamed for &DebugVar<'_, E> {}

/// Prints a formatted message using the print debug layer in Vulkan, or `printf` in CUDA.
pub fn printf_expand(scope: &Scope, format_string: impl Into<String>, args: Vec<Value>) {
    scope.register(&PrintfOp::new(scope.ctx_mut(), format_string.into(), args));
}

/// Print a formatted message using the target's debug print facilities. The format string is target
/// specific, but Vulkan and CUDA both use the C++ conventions. WGSL isn't currently supported.
#[macro_export]
macro_rules! debug_print {
    ($format:literal, $($args:expr),*) => {
        {
            let _ = $format;
            $(let _ = $args;)*
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::debug_print!($format, $($args),*);
    };
}

/// Print a formatted message using the target's debug print facilities. The format string is target
/// specific, but Vulkan and CUDA both use the C++ conventions. WGSL isn't currently supported.
#[macro_export]
macro_rules! __expand_debug_print {
    ($scope:expr, $format:expr, $($args:expr),*) => {
        {
            let args = $crate::__private::vec![$($crate::ir::ExpandValue::from($args).read_value($scope)),*];
            $crate::frontend::printf_expand($scope, $format, args);
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::__expand_debug_print!($format, $($args),*)
    };
}

pub mod cube_comment {
    use alloc::string::ToString;

    use cubecl_ir::{Scope, dialect::general::CommentOp};

    pub fn expand(scope: &Scope, content: &str) {
        scope.register(&CommentOp::new(scope.ctx_mut(), content.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use crate as cubecl;
    use crate::prelude::*;
    use alloc::vec::Vec;
    use cubecl_ir::{
        pliron::{
            linked_list::ContainsLinkedList,
            location::{Located, Location},
        },
        settings::{DebugInfo, Dim3, ExecutionMode, KernelSettings},
    };

    /// The line where the frame of `double` opens, and the line of its statement.
    const DOUBLE_LINES: [u32; 2] = [line!() + 1, line!() + 3];
    #[cube]
    fn double(x: u32) -> u32 {
        x + x
    }

    #[cube]
    fn quadruple(x: u32) -> u32 {
        double(x) + double(x)
    }

    #[cube]
    fn add_double(x: u32) -> u32 {
        x + double(x)
    }

    const TWO_SUMS_LINE: u32 = line!() + 3;
    #[cube]
    fn two_sums(x: u32) -> u32 {
        (x + 1) * (x + 2)
    }

    /// The line of the statement, and the line of the call that continues it.
    const SPLIT_LINES: [u32; 2] = [line!() + 4, line!() + 5];
    #[cube]
    #[rustfmt::skip]
    fn split_double(x: u32) -> u32 {
        x
            + double(x)
    }

    /// A `#[cube]` function from a `macro_rules!` body: the operator comes from the invocation,
    /// so its tokens have another hygiene than the code of the macro.
    macro_rules! unary_cube_fn {
        ($name:ident, $op:tt) => {
            #[cube]
            fn $name(x: u32) -> u32 {
                $op(x + 1)
            }
        };
    }

    unary_cube_fn!(not_after_one, !);

    #[cube(no_debug_symbols)]
    fn plain_double(x: u32) -> u32 {
        x + x
    }

    const CALL_PLAIN_LINE: u32 = line!() + 3;
    #[cube]
    fn call_plain(x: u32) -> u32 {
        plain_double(x)
    }

    #[derive(CubeType, Clone, Copy)]
    struct Pair {
        a: u32,
    }

    const SUM_LINES: [u32; 2] = [line!() + 4, line!() + 5];
    #[cube]
    impl Pair {
        fn sum(self, x: u32) -> u32 {
            let y = self.a + x;
            y * x
        }
    }

    #[cube]
    fn use_pair(x: u32) -> u32 {
        let pair = Pair { a: x };
        pair.sum(x)
    }

    /// A plain Rust type: it has no `CubeType` and no `CubeDebug`.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sign {
        Plus,
        Minus,
    }

    #[cube]
    fn pick(#[comptime] sign: Sign) -> comptime_type!(Sign) {
        sign
    }

    #[cube]
    fn signed(x: u32, #[comptime] sign: Sign) -> u32 {
        let sign = pick(sign);
        if comptime!(sign == Sign::Minus) {
            x * x
        } else {
            x + x
        }
    }

    /// Runs `expand` on a new kernel scope and returns the location of each op it inserted.
    fn locations(
        debug_info: DebugInfo,
        expand: impl FnOnce(&Scope, NativeExpand<u32>),
    ) -> Vec<Location> {
        let settings =
            KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32)
                .debug_info(debug_info);
        let scope = &Scope::root(settings);
        let x = NativeExpand::<u32>::from_lit(scope, 2);
        expand(scope, x);

        let ctx = scope.ctx();
        let block = scope.state().entry_func.get_entry_block(ctx);
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        ops.into_iter().map(|op| op.deref(ctx).loc()).collect()
    }

    /// A function name, and a line and a column in the function.
    type Frame<'a> = (&'a str, u32, u32);

    fn frame(loc: &Location) -> Option<Frame<'_>> {
        match loc {
            Location::Named { name, child_loc } => match child_loc.as_ref() {
                Location::SrcPos { pos, .. } => Some((
                    name,
                    u32::try_from(pos.line).ok()?,
                    u32::try_from(pos.column).ok()?,
                )),
                _ => None,
            },
            _ => None,
        }
    }

    /// The innermost frame and its caller, for each op that a call inserted.
    fn call_sites(locations: &[Location]) -> Vec<(Frame<'_>, Frame<'_>)> {
        locations
            .iter()
            .filter_map(|loc| match loc {
                Location::CallSite { callee, caller } => Some((frame(callee)?, frame(caller)?)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn ops_get_the_line_of_their_expression() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            double::expand(scope, x);
        });
        let frames = locations.iter().filter_map(frame).collect::<Vec<_>>();
        assert!(
            frames.iter().any(|(_, line, _)| *line == DOUBLE_LINES[1]),
            "{frames:?}"
        );
        for (name, line, _) in frames {
            assert_eq!(name, "double");
            assert!(DOUBLE_LINES.contains(&line), "{line}");
        }
    }

    #[test]
    fn inlined_calls_keep_their_call_site() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            quadruple::expand(scope, x);
        });
        let call_sites = call_sites(&locations);
        // Every op of the two `double` calls records that `quadruple` called it.
        assert!(!call_sites.is_empty(), "{locations:?}");
        for ((inner, line, _), (outer, ..)) in call_sites {
            assert_eq!(inner, "double");
            assert!(DOUBLE_LINES.contains(&line), "{line}");
            assert_eq!(outer, "quadruple");
        }
    }

    /// The addition after `double(x)` is at the start of `x + double(x)`, not at the call.
    #[test]
    fn an_op_after_a_call_gets_the_position_of_its_expression() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            add_double::expand(scope, x);
        });
        let (_, (_, call_line, call_column)) = call_sites(&locations)[0];
        let (name, line, column) = locations.last().and_then(frame).expect("the addition");
        assert_eq!((name, line), ("add_double", call_line));
        assert!(column < call_column, "{column} {call_column}");
    }

    /// Each op on one line gets the column of its own expression: `x + 1`, `x + 2`, and the
    /// product, which starts first.
    #[test]
    fn ops_on_one_line_get_the_column_of_their_expression() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            two_sums::expand(scope, x);
        });
        let mut columns = locations
            .iter()
            .filter_map(frame)
            .filter(|(_, line, _)| *line == TWO_SUMS_LINE)
            .map(|(.., column)| column)
            .collect::<Vec<_>>();
        let (.., product) = locations.last().and_then(frame).expect("the product");
        columns.sort_unstable();
        columns.dedup();
        assert_eq!(columns.len(), 3, "{locations:?}");
        assert_eq!(product, columns[0], "{locations:?}");
    }

    /// A call on another line than its statement records that line as its call site. The op
    /// after the call gets the line of the statement back.
    #[test]
    fn a_call_on_a_continuation_line_keeps_its_line() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            split_double::expand(scope, x);
        });
        let call_sites = call_sites(&locations);
        assert!(!call_sites.is_empty(), "{locations:?}");
        for (_, (outer, line, _)) in call_sites {
            assert_eq!((outer, line), ("split_double", SPLIT_LINES[1]));
        }
        let (name, line, _) = locations.last().and_then(frame).expect("the addition");
        assert_eq!((name, line), ("split_double", SPLIT_LINES[0]));
    }

    /// The debug code that the macro adds must resolve `scope` also for tokens from a
    /// `macro_rules!` invocation.
    #[test]
    fn a_function_from_macro_rules_gets_locations() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            not_after_one::expand(scope, x);
        });
        assert!(
            locations
                .iter()
                .filter_map(frame)
                .any(|(name, ..)| name == "not_after_one"),
            "{locations:?}"
        );
    }

    /// A function with `no_debug_symbols` opens no frame. Its ops get the location of the call.
    #[test]
    fn a_function_without_debug_symbols_has_the_location_of_its_call() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            call_plain::expand(scope, x);
        });
        assert!(call_sites(&locations).is_empty(), "{locations:?}");
        let last = locations.last().and_then(frame);
        assert_eq!(
            last.map(|(name, line, _)| (name, line)),
            Some(("call_plain", CALL_PLAIN_LINE))
        );
    }

    /// A method of a `#[cube] impl` is one frame, with the lines of its statements.
    #[test]
    fn a_method_is_one_frame() {
        let locations = locations(DebugInfo::LineTables, |scope, x| {
            use_pair::expand(scope, x);
        });
        let call_sites = call_sites(&locations);
        for line in SUM_LINES {
            assert!(
                call_sites
                    .iter()
                    .any(|((inner, at, _), (outer, ..))| *inner == "Pair :: sum"
                        && *at == line
                        && *outer == "use_pair"),
                "no op at line {line}: {call_sites:?}"
            );
        }
    }

    /// A variable of a plain Rust type, such as a comptime enum, needs no `CubeDebug`. It gets no
    /// name, and the ops of the function still get their lines.
    #[test]
    fn a_plain_variable_needs_no_cube_debug() {
        for sign in [Sign::Plus, Sign::Minus] {
            let locations = locations(DebugInfo::LineTables, |scope, x| {
                signed::expand(scope, x, sign);
            });
            assert!(
                locations
                    .iter()
                    .filter_map(frame)
                    .any(|(name, ..)| name == "signed"),
                "{locations:?}"
            );
        }
    }

    #[cube]
    fn add_one(x: u32) -> u32 {
        x + 1
    }

    #[cube]
    fn sum_indices(n: u32) -> u32 {
        let mut acc = 0u32;
        for i in 0..n {
            acc += add_one(i);
        }
        acc
    }

    /// The loop index is a block argument and has no defining op. Debug data names the parameter
    /// `x` of `add_one`, so the name must leave such a value as it is.
    #[test]
    fn a_block_argument_keeps_its_name() {
        locations(DebugInfo::Full, |scope, x| {
            sum_indices::expand(scope, x);
        });
    }

    #[test]
    fn no_locations_without_debug_info() {
        let locations = locations(DebugInfo::None, |scope, x| {
            double::expand(scope, x);
        });
        assert!(locations.iter().all(Location::is_unknown));
    }
}
