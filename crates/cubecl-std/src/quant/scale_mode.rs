use cubecl::prelude::*;
use cubecl_core as cubecl;

/// How a [`QuantizedView`](crate::quant::view::QuantizedView) resolves the scale of each read.
/// The discriminant is comptime, so each mode compiles its own kernel with nothing of the others
/// in it.
#[derive(Clone, Copy, CubeType)]
pub enum ScaleMode {
    /// Read one scale per position through the scales view.
    Addressed,
    /// Read one scale per position, folded with this register: the product of every outer level's
    /// scale, each covering the whole tensor, read once per kernel from their bindings.
    Folded(f32),
    /// This register is the whole scale for every value the view serves, whatever the caller
    /// folded into it. The scales view is never read: its address arithmetic and its load leave
    /// the kernel.
    Uniform(f32),
}

impl Clone for ScaleModeExpand {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for ScaleModeExpand {}

/// Mirrors [`ComptimeOption`]'s launch story: the discriminant is a compilation fact, the payload
/// a scalar argument.
pub enum ScaleModeArgs<R: Runtime> {
    Addressed,
    Folded(<f32 as LaunchArg>::RuntimeArg<R>),
    Uniform(<f32 as LaunchArg>::RuntimeArg<R>),
}

/// Only the discriminant: a scalar's registration carries no compilation state of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleModeCompilationArg {
    Addressed,
    Folded,
    Uniform,
}

impl LaunchArg for ScaleMode {
    type RuntimeArg<R: Runtime> = ScaleModeArgs<R>;
    type CompilationArg = ScaleModeCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        match arg {
            ScaleModeArgs::Addressed => ScaleModeCompilationArg::Addressed,
            ScaleModeArgs::Folded(scale) => {
                <f32 as LaunchArg>::register(scale, launcher);
                ScaleModeCompilationArg::Folded
            }
            ScaleModeArgs::Uniform(scale) => {
                <f32 as LaunchArg>::register(scale, launcher);
                ScaleModeCompilationArg::Uniform
            }
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> ScaleModeExpand {
        match arg {
            ScaleModeCompilationArg::Addressed => ScaleModeExpand::Addressed,
            ScaleModeCompilationArg::Folded => {
                ScaleModeExpand::Folded(<f32 as LaunchArg>::expand(&(), builder))
            }
            ScaleModeCompilationArg::Uniform => {
                ScaleModeExpand::Uniform(<f32 as LaunchArg>::expand(&(), builder))
            }
        }
    }
}
