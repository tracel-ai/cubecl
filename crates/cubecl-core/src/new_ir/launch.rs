use crate::{prelude::ArgSettings, Runtime};

use super::{compute::KernelBuilder, GlobalVariable, SquareType};

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
