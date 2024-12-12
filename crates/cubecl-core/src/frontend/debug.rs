use crate::ir::{DebugInfo, Variable};

use super::CubeContext;

/// Calls a function and inserts debug symbols if debug is enabled.
pub fn debug_call_expand<C>(
    context: &mut CubeContext,
    name: &'static str,
    call: impl FnOnce(&mut CubeContext) -> C,
) -> C {
    if context.debug_enabled {
        context.register(DebugInfo::BeginCall {
            name: name.to_string(),
        });

        let ret = call(context);

        context.register(DebugInfo::EndCall);

        ret
    } else {
        call(context)
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
