use crate::ir::{NonSemantic, Scope, Variable};

/// Calls a function and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn debug_call_expand<C>(
    scope: &mut Scope,
    name: &'static str,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut Scope) -> C,
) -> C {
    if scope.debug_enabled {
        scope.register(NonSemantic::BeginCall {
            name: name.to_string(),
            line,
            col,
        });

        let ret = call(scope);

        scope.register(NonSemantic::EndCall);

        ret
    } else {
        call(scope)
    }
}

/// Calls an intrinsic op and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn spanned_expand<C>(
    scope: &mut Scope,
    line: u32,
    col: u32,
    call: impl FnOnce(&mut Scope) -> C,
) -> C {
    if scope.debug_enabled {
        scope.register(NonSemantic::Line { line, col });
        call(scope)
    } else {
        call(scope)
    }
}

/// Adds source instruction if debug is enabled
#[track_caller]
pub fn debug_source_expand(scope: &mut Scope, name: &str, file: &str, line: u32, col: u32) {
    if scope.debug_enabled {
        // Normalize to linux separators
        let file = file.replace("\\", "/");
        scope.register(NonSemantic::Source {
            name: name.into(),
            file_name: format!("./{file}"),
            line,
            col,
        });
    }
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
    ($scope:expr, $format:literal, $($args:expr),*) => {
        {
            let args = vec![$(*$args.expand),*];
            $crate::frontend::printf_expand($scope, $format, args);
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::debug_print_expand!($format, $($args),*)
    };
}
