use crate::{
    compute::{KernelBuilder, KernelLauncher},
    prelude::{ArgSettings, CompilationArg, LaunchArg, LaunchArgExpand},
    Runtime,
};
use std::marker::PhantomData;

impl<C: CompilationArg> CompilationArg for Option<C> {}

pub struct CubeOptionHandleRef<'a, H, R: Runtime> {
    pub inner: Option<&'a H>,
    runtime: PhantomData<R>,
}

impl<T: LaunchArgExpand> LaunchArgExpand for Option<T> {
    type CompilationArg = Option<T::CompilationArg>;

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Option<T::ExpandType> {
        arg.as_ref().map(|arg| T::expand(arg, builder))
    }

    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> Option<T::ExpandType> {
        arg.as_ref().map(|arg| T::expand_output(arg, builder))
    }
}

impl<R: Runtime, A: ArgSettings<R>> ArgSettings<R> for Option<A> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let Some(arg) = self {
            arg.register(launcher);
        }
    }
}

impl<T: LaunchArg> LaunchArg for Option<T> {
    type RuntimeArg<'a, R: Runtime> = Option<T::RuntimeArg<'a, R>>;
    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        runtime_arg.as_ref().map(T::compilation_arg)
    }
}
