use core::fmt::Display;

use alloc::{string::String, vec::Vec};

use crate::TypeHash;

use crate::{fmt_vararg, OperationCode, OperationReflect};

use super::Variable;

/// Operations that don't change the semantics of the kernel. In other words, operations that do not
/// perform any computation, if they run at all. i.e. `println`, comments and debug symbols.
///
/// Can be safely removed or ignored without changing the kernel result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = NonSemanticOpCode)]
pub enum NonSemantic {
    Source {
        name: String,
        file_name: String,
        line: u32,
        col: u32,
    },
    BeginCall {
        name: String,
        line: u32,
        col: u32,
    },
    EndCall,
    Line {
        line: u32,
        col: u32,
    },
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
    Comment {
        content: String,
    },
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
                write!(f, "print({format_string}, {})", fmt_vararg(args))
            }
            NonSemantic::Comment { content } => write!(f, "//{content}"),
            _ => {
                // Debug info has no semantic meaning
                Ok(())
            }
        }
    }
}
