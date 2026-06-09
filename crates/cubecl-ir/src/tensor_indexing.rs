use crate::{ClampMode, TypeHash};
use crate::{OperationArgs, OperationCode};
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt::Display;

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = TensorIndexingOpCode)]
/// Operations available on a barrier
pub enum TensorIndexingOps {
    CreateLayout {
        shape: Vec<Variable>,
        strides: Option<Vec<Variable>>,
        clamp_mode: ClampMode,
    },
    CreateView,
    Slice {
        layout: Variable,
        offsets: Vec<Variable>,
        shape: Vec<Variable>,
    },
}

impl OperationReflect for TensorIndexingOps {
    type OpCode = TensorIndexingOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }

    fn args(&self) -> Option<Vec<Variable>> {
        match self {
            TensorIndexingOps::CreateLayout { .. }
            | TensorIndexingOps::CreateView
            | TensorIndexingOps::Slice { .. } => None,
        }
    }

    fn sanitize_args(&mut self, scope: &crate::Scope) {
        match self {
            TensorIndexingOps::CreateLayout { shape, strides, .. } => {
                shape.sanitize_args_ptr(scope);
                strides.sanitize_args_ptr(scope);
            }
            TensorIndexingOps::CreateView => {}
            TensorIndexingOps::Slice { offsets, shape, .. } => {
                offsets.sanitize_args_ptr(scope);
                shape.sanitize_args_ptr(scope);
            }
        }
    }
}

impl Display for TensorIndexingOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorIndexingOps::CreateLayout {
                shape,
                strides,
                clamp_mode,
            } => {
                let shape: Vec<_> = shape.iter().map(|it| it.to_string()).collect();
                let strides: Option<Vec<_>> = strides
                    .as_ref()
                    .map(|strides| strides.iter().map(|it| it.to_string()).collect());
                let strides = strides
                    .map(|strides| alloc::format!("[{}]", strides.join(", ")))
                    .unwrap_or("None".into());
                write!(
                    f,
                    "create_layout([{}], strides: {strides}, clamp_mode: {clamp_mode:?}",
                    shape.join(", ")
                )
            }
            TensorIndexingOps::CreateView => {
                write!(f, "create_view()")
            }
            TensorIndexingOps::Slice {
                layout,
                offsets,
                shape,
            } => {
                let offsets = offsets.iter().map(|it| it.to_string()).collect::<Vec<_>>();
                let shape = shape.iter().map(|it| it.to_string()).collect::<Vec<_>>();
                write!(
                    f,
                    "slice({layout}, offsets: [{}], shape: [{}])",
                    offsets.join(", "),
                    shape.join(", ")
                )
            }
        }
    }
}
