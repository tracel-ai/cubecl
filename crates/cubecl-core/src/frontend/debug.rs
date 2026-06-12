use alloc::{string::String, vec::Vec};
use cubecl_ir::{dialect::general::PrintfOp, pliron::value::Value};

use crate::ir::Scope;

use super::CubeDebug;

/// Calls a function and inserts debug symbols if debug is enabled.
#[track_caller]
pub fn debug_call_expand<C>(
    scope: &Scope,
    _line: u32,
    _col: u32,
    call: impl FnOnce(&Scope) -> C,
) -> C {
    // Save source_loc before the call so it can be restored once the call returns
    // let source_loc = scope.debug.source_loc.take();
    // scope.update_span(line, col);
    // scope.register(NonSemantic::EnterDebugScope);
    let ret = call(scope);
    // scope.register(NonSemantic::ExitDebugScope);
    // *scope.debug.source_loc.borrow_mut() = source_loc;
    ret
}

/// Adds source instruction if debug is enabled
#[track_caller]
pub fn debug_source_expand(
    _scope: &Scope,
    _name: &'static str,
    _file: &'static str,
    _source_text: &'static str,
    _line: u32,
    _column: u32,
) {
    // let file = file.replace("\\", "/");
    // scope.update_source(CubeFnSource {
    //     function_name: name.into(),
    //     file: file.into(),
    //     source_text: source_text.into(),
    //     line,
    //     column,
    // });
}

/// Registers name for an expand if possible
#[track_caller]
pub fn debug_var_expand<E: CubeDebug>(scope: &Scope, name: &'static str, expand: E) -> E {
    expand.set_debug_name(scope, name);
    expand
}

/// Prints a formatted message using the print debug layer in Vulkan, or `printf` in CUDA.
pub fn printf_expand(scope: &Scope, format_string: impl Into<String>, args: Vec<Value>) {
    scope.register(&PrintfOp::new(
        &mut scope.ctx_mut(),
        format_string.into(),
        args,
    ));
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
        scope.register(&CommentOp::new(
            &mut scope.ctx_mut(),
            content.to_string().into(),
        ));
    }
}
