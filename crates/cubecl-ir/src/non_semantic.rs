use core::fmt::Display;

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::TypeHash;

use crate::{OperationCode, OperationReflect, fmt_vararg};

use super::Variable;

/// Operations that don't change the semantics of the kernel. In other words, operations that do not
/// perform any computation, if they run at all. i.e. `println`, comments and debug symbols.
///
/// Can be safely removed or ignored without changing the kernel result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = NonSemanticOpCode)]
pub enum NonSemantic {
    /// Enter a new debug scope, this happens recursively when a cube function is called and
    /// inlined into the kernel. Should push a new frame onto the debug call stack.
    EnterDebugScope,
    /// Exit a debug scope and resume execution at the previous stack frame.
    ExitDebugScope,
    /// Print a formatted string with arguments to the backend's debugging facilit
    /// (i.e. validation layer). The syntax of the format string depends on the backend, but tends
    /// to follow C++ convention.
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
    /// Insert a comment into the compiled source
    Comment { content: String },
}

impl OperationReflect for NonSemantic {
    type OpCode = NonSemanticOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }
}

impl Display for NonSemantic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NonSemantic::Print {
                format_string,
                args,
            } => {
                write!(f, "print({format_string:?}, {})", fmt_vararg(args))
            }
            NonSemantic::Comment { content } => write!(f, "//{content}"),
            // Scopes don't have meaning to the user
            _ => Ok(()),
        }
    }
}

/// A Rust source location, including the file, line and column
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct SourceLoc {
    pub line: u32,
    pub column: u32,
    pub source: CubeFnSource,
}

/// A cube function's source
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct CubeFnSource {
    pub function_name: Cow<'static, str>,
    pub file: Cow<'static, str>,
    pub source_text: Cow<'static, str>,
    pub line: u32,
    pub column: u32,
}
