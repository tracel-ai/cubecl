use crate::{self as cubecl, ExpandType};
use cubecl::prelude::*;

pub enum OptionExpand<T: CubeType> {
    Some(<T as CubeType>::ExpandType),
    None,
}

impl<T: CubeType> CubeType for Option<T> {
    type ExpandType = OptionExpand<T>;
}

impl<T: CubeType> IntoMut for OptionExpand<T> {
    fn into_mut(self, scope: &mut cubecl::prelude::Scope) -> Self {
        match self {
            OptionExpand::Some(arg_0) => OptionExpand::Some(IntoMut::into_mut(arg_0, scope)),
            OptionExpand::None => OptionExpand::None,
        }
    }
}

impl<T: CubeType> CubeDebug for Option<T> {}
impl<T: CubeType> CubeDebug for OptionExpand<T> {}

impl<T: CubeType> Clone for OptionExpand<T> {
    fn clone(&self) -> Self {
        match self {
            OptionExpand::Some(arg_0) => OptionExpand::Some(arg_0.clone()),
            OptionExpand::None => OptionExpand::None,
        }
    }
}

#[allow(non_snake_case)]
pub trait CubeOption<T: CubeType> {
    fn new_Some(_0: T) -> Option<T> {
        Option::Some(_0)
    }
    fn new_None() -> Option<T> {
        Option::None
    }
    fn __expand_Some(_scope: &mut Scope, _0: ExpandType<T>) -> OptionExpand<T> {
        OptionExpand::Some(_0)
    }
    fn __expand_new_Some(_scope: &mut Scope, _0: ExpandType<T>) -> OptionExpand<T> {
        OptionExpand::Some(_0)
    }
    fn __expand_new_None(_scope: &mut Scope) -> OptionExpand<T> {
        OptionExpand::None
    }
}

impl<T: CubeType> CubeOption<T> for Option<T> {}

impl<T: CubeType> OptionExpand<T> {
    pub fn is_some(&self) -> bool {
        match self {
            OptionExpand::Some(_) => true,
            OptionExpand::None => false,
        }
    }

    pub fn unwrap(self) -> T::ExpandType {
        match self {
            Self::Some(val) => val,
            Self::None => panic!("Unwrap on a None CubeOption"),
        }
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn unwrap_or(self, fallback: T::ExpandType) -> T::ExpandType {
        match self {
            OptionExpand::Some(val) => val,
            OptionExpand::None => fallback,
        }
    }

    pub fn __expand_is_some_method(&self, _scope: &mut Scope) -> bool {
        self.is_some()
    }

    pub fn __expand_unwrap_method(self, _scope: &mut Scope) -> T::ExpandType {
        self.unwrap()
    }

    pub fn __expand_is_none_method(&self, _scope: &mut Scope) -> bool {
        self.is_none()
    }

    pub fn __expand_unwrap_or_method(
        self,
        _scope: &mut Scope,
        fallback: T::ExpandType,
    ) -> T::ExpandType {
        self.unwrap_or(fallback)
    }

    // Expanded types are just cloned
    pub fn __expand_as_ref_method(self, _scope: &mut Scope) -> Self {
        self
    }

    pub fn __expand_map_method<R: CubeType>(
        self,
        scope: &mut Scope,
        transform: impl FnOnce(&mut Scope, T::ExpandType) -> R::ExpandType,
    ) -> OptionExpand<R> {
        match self {
            OptionExpand::Some(value) => OptionExpand::Some(transform(scope, value)),
            OptionExpand::None => OptionExpand::None,
        }
    }

    pub fn __expand_unwrap_or_else_method(
        self,
        scope: &mut Scope,
        or_else: impl FnOnce(&mut Scope) -> T::ExpandType,
    ) -> T::ExpandType {
        match self {
            OptionExpand::Some(value) => value,
            OptionExpand::None => or_else(scope),
        }
    }
}

pub enum OptionArgs<'a, T: LaunchArg, R: Runtime> {
    Some(<T as LaunchArg>::RuntimeArg<'a, R>),
    None,
}

impl<'a, T: LaunchArg, R: Runtime> From<Option<<T as LaunchArg>::RuntimeArg<'a, R>>>
    for OptionArgs<'a, T, R>
{
    fn from(value: Option<<T as LaunchArg>::RuntimeArg<'a, R>>) -> Self {
        match value {
            Some(arg) => Self::Some(arg),
            None => Self::None,
        }
    }
}

impl<T: LaunchArg, R: Runtime> ArgSettings<R> for OptionArgs<'_, T, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        match self {
            OptionArgs::Some(arg) => {
                arg.register(launcher);
            }
            OptionArgs::None => {}
        }
    }
}
impl<T: LaunchArg> LaunchArg for Option<T> {
    type RuntimeArg<'a, R: Runtime> = OptionArgs<'a, T, R>;
    type CompilationArg = OptionCompilationArg<T>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            OptionArgs::Some(arg) => OptionCompilationArg::Some(T::compilation_arg(arg)),
            OptionArgs::None => OptionCompilationArg::None,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            OptionCompilationArg::Some(arg) => OptionExpand::Some(T::expand(arg, builder)),
            OptionCompilationArg::None => OptionExpand::None,
        }
    }

    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            OptionCompilationArg::Some(arg) => OptionExpand::Some(T::expand_output(arg, builder)),
            OptionCompilationArg::None => OptionExpand::None,
        }
    }
}

pub enum OptionCompilationArg<T: LaunchArg> {
    Some(<T as LaunchArg>::CompilationArg),
    None,
}

impl<T: LaunchArg> Clone for OptionCompilationArg<T> {
    fn clone(&self) -> Self {
        match self {
            OptionCompilationArg::Some(arg) => OptionCompilationArg::Some(arg.clone()),
            OptionCompilationArg::None => OptionCompilationArg::None,
        }
    }
}

impl<T: LaunchArg> PartialEq for OptionCompilationArg<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OptionCompilationArg::Some(arg_0), OptionCompilationArg::Some(arg_1)) => {
                arg_0 == arg_1
            }
            (OptionCompilationArg::None, OptionCompilationArg::None) => true,
            _ => false,
        }
    }
}

impl<T: LaunchArg> Eq for OptionCompilationArg<T> {}

impl<T: LaunchArg> core::hash::Hash for OptionCompilationArg<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            OptionCompilationArg::Some(arg) => {
                arg.hash(state);
            }
            OptionCompilationArg::None => {}
        };
    }
}

impl<T: LaunchArg> core::fmt::Debug for OptionCompilationArg<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OptionCompilationArg::Some(arg) => f.debug_tuple("Some").field(arg).finish(),
            OptionCompilationArg::None => write!(f, "None"),
        }
    }
}

impl<T: LaunchArg> CompilationArg for OptionCompilationArg<T> {}
