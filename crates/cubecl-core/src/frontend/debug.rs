use cubecl_ir::CubeFnSource;

use crate::ir::{NonSemantic, Scope, Variable};

use super::CubeDebug;

/// Calls a function and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn debug_call_expand<C>(
    scope: &mut Scope,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut Scope) -> C,
) -> C {
    // Save source_loc before the call so it can be restored once the call returns
    let source_loc = scope.debug.source_loc.take();
    scope.update_span(line, col);
    scope.register(NonSemantic::EnterDebugScope);
    let ret = call(scope);
    scope.register(NonSemantic::ExitDebugScope);
    scope.debug.source_loc = source_loc;
    ret
}

/// Calls an intrinsic op and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn spanned_expand<C>(
    scope: &mut Scope,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut Scope) -> C,
) -> C {
    scope.update_span(line, col);
    call(scope)
}

/// Adds source instruction if debug is enabled
#[track_caller]
pub fn debug_source_expand(
    scope: &mut Scope,
    name: &'static str,
    file: &'static str,
    source_text: &'static str,
    line: u32,
    column: u32,
) {
    let file = file.replace("\\", "/");
    scope.update_source(CubeFnSource {
        function_name: name.into(),
        file: file.into(),
        source_text: source_text.into(),
        line,
        column,
    });
}

/// Registers name for an expand if possible
#[track_caller]
#[inline(never)]
pub fn debug_var_expand<E: CubeDebug>(scope: &mut Scope, name: &'static str, expand: E) -> E {
    expand.set_debug_name(scope, name);
    expand
}

/// Prints a formatted message using the print debug layer in Vulkan, or `printf` in CUDA.
pub fn printf_expand(scope: &mut Scope, format_string: impl Into<String>, args: Vec<Variable>) {
    scope.register(NonSemantic::Print {
        format_string: format_string.into(),
        args,
    });
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
macro_rules! debug_print_expand {
    ($scope:expr, $format:expr, $($args:expr),*) => {
        {
            let args = vec![$(*$crate::ir::ExpandElement::from($args)),*];
            $crate::frontend::printf_expand($scope, $format, args);
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::debug_print_expand!($format, $($args),*)
    };
}
