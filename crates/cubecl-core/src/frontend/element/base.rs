use crate::{
    ir::Variable,
    new_ir::{GlobalVariable, SquareType},
    prelude::{KernelBuilder, KernelLauncher},
    KernelSettings, Runtime,
};
use alloc::rc::Rc;
use std::collections::HashMap;

/// Defines how a [launch argument](LaunchArg) can be expanded.
///
/// Normally this type should be implemented two times for an argument.
/// Once for the reference and the other for the mutable reference. Often time, the reference
/// should expand the argument as an input while the mutable reference should expand the argument
/// as an output.
pub trait LaunchArgExpand: SquareType + Sized {
    /// Register an input variable during compilation that fill the [KernelBuilder].
    fn expand(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self>;
    /// Register an output variable during compilation that fill the [KernelBuilder].
    fn expand_output(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        Self::expand(builder, vectorization)
    }
}

/// Defines a type that can be used as argument to a kernel.
pub trait LaunchArg: LaunchArgExpand + Send + Sync + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;
}

pub type RuntimeArg<'a, T, R> = <T as LaunchArg>::RuntimeArg<'a, R>;

impl<R: Runtime> ArgSettings<R> for () {
    fn register(&self, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }
}

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
    /// Configure an input argument at the given position.
    fn configure_input(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
    /// Configure an output argument at the given position.
    fn configure_output(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
}

/// Reference to a JIT variable
#[derive(Clone, Debug, PartialEq)]
pub enum ExpandElement {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
    /// Struct with subexpressions
    Struct(HashMap<&'static str, ExpandElement>),
}

/// Weak reference to a JIT variable for variable name mapping
#[derive(Clone, Debug, PartialEq)]
pub enum ExpandElementWeak {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
    /// Struct with subexpressions
    Struct(HashMap<&'static str, ExpandElement>),
}

// impl PartialEq for ExpandElementWeak {
//     fn eq(&self, other: &Self) -> bool {
//         match (self, other) {
//             (ExpandElementWeak::Managed(var), ExpandElementWeak::Managed(var2)) => var
//                 .upgrade()
//                 .zip(var2.upgrade())
//                 .map(|(var1, var2)| var1 == var2)
//                 .unwrap_or(false),
//             (ExpandElementWeak::Plain(var), ExpandElementWeak::Plain(var2)) => var == var2,
//             _unused => false,
//         }
//     }
// }

impl ExpandElementWeak {
    pub fn upgrade(self) -> Option<ExpandElement> {
        match self {
            ExpandElementWeak::Managed(var) => Some(ExpandElement::Managed(var)),
            ExpandElementWeak::Plain(var) => Some(ExpandElement::Plain(var)),
            ExpandElementWeak::Struct(vars) => Some(ExpandElement::Struct(vars)),
        }
    }
}

impl ExpandElement {
    /// If the element can be mutated inplace, potentially reusing the register.
    pub fn can_mut(&self) -> bool {
        match self {
            ExpandElement::Managed(var) => {
                if let Variable::Local { .. } = var.as_ref() {
                    Rc::strong_count(var) <= 2
                } else {
                    false
                }
            }
            ExpandElement::Plain(Variable::LocalArray { .. } | Variable::SharedMemory { .. }) => {
                true
            }
            _ => false,
        }
    }

    pub fn as_weak(&self) -> ExpandElementWeak {
        match self {
            ExpandElement::Managed(var) => ExpandElementWeak::Managed(var.clone()),
            ExpandElement::Plain(var) => ExpandElementWeak::Plain(*var),
            ExpandElement::Struct(var) => ExpandElementWeak::Struct(var.clone()),
        }
    }

    pub fn into_variable(self) -> Variable {
        match self {
            ExpandElement::Managed(var) => *var,
            ExpandElement::Plain(var) => var,
            ExpandElement::Struct(_) => panic!("Can't turn struct into variable"),
        }
    }

    pub fn as_variable(&self) -> Variable {
        match self {
            ExpandElement::Managed(var) => *var.as_ref(),
            ExpandElement::Plain(var) => *var,
            ExpandElement::Struct(_) => panic!("Can't turn struct into variable"),
        }
    }

    pub fn item(&self) -> crate::ir::Item {
        self.as_variable().item()
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        match value {
            ExpandElement::Managed(var) => *var,
            ExpandElement::Plain(var) => var,
            ExpandElement::Struct(_) => panic!("Can't turn struct into variable"),
        }
    }
}
