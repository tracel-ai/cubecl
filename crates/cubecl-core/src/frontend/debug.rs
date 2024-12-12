use crate::ir::{DebugInfo, Variable};

use super::CubeContext;

/// Calls a function and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn debug_call_expand<C>(
    context: &mut CubeContext,
    name: &'static str,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut CubeContext) -> C,
) -> C {
    if context.debug_enabled {
        context.register(DebugInfo::BeginCall {
            name: name.to_string(),
            line,
            col,
        });

        let ret = call(context);

        context.register(DebugInfo::EndCall);

        ret
    } else {
        call(context)
    }
}

/// Calls an intrinsic op and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn spanned_expand<C>(
    context: &mut CubeContext,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut CubeContext) -> C,
) -> C {
    if context.debug_enabled {
        context.register(DebugInfo::Span { line, col });
        call(context)
    } else {
        call(context)
    }
}

/// Adds source instruction if debug is enabled
#[track_caller]
pub fn debug_source_expand(context: &mut CubeContext, name: &str, file: &str, line: u32, col: u32) {
    if context.debug_enabled {
        // Normalize to linux separators
        let file = file.replace("\\", "/");
        context.register(DebugInfo::Source {
            name: name.into(),
            file_name: format!("./{file}"),
            line,
            col,
        });
    }
}

/// Prints a formatted message using the print debug layer in Vulkan, or `printf` in CUDA.
pub fn printf_expand(
    context: &mut CubeContext,
    format_string: impl Into<String>,
    args: Vec<Variable>,
) {
    context.register(DebugInfo::Print {
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
    ($context:expr, $format:literal, $($args:expr),*) => {
        {
            let args = vec![$(*$args.expand),*];
            $crate::frontend::printf_expand($context, $format, args);
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::debug_print_expand!($format, $($args),*)
    };
}
