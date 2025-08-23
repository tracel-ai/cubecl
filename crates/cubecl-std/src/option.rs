use cubecl::prelude::*;
use cubecl_core as cubecl;
use serde::{Deserialize, Serialize};

#[derive(CubeType, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq, Debug)]
pub enum CubeOption<T: CubeType> {
    Some(T),
    None,
}

#[cube]
impl<T: CubeType> CubeOption<T> {
    pub fn is_some(&self) -> bool {
        match self {
            CubeOption::Some(_) => true,
            CubeOption::None => false,
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            CubeOption::Some(val) => val,
            CubeOption::None => panic!("Unwrap on a None CubeOption"),
        }
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn unwrap_or(self, fallback: T) -> T {
        match self {
            CubeOption::Some(val) => val,
            CubeOption::None => fallback,
        }
    }
}

impl<T: CubeType> CubeOptionExpand<T> {
    pub fn is_some(&self) -> bool {
        match self {
            CubeOptionExpand::Some(_) => true,
            CubeOptionExpand::None => false,
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
            CubeOptionExpand::Some(val) => val,
            CubeOptionExpand::None => fallback,
        }
    }
}

impl<T: CubeType + Into<T::ExpandType>> From<CubeOption<T>> for CubeOptionExpand<T> {
    fn from(value: CubeOption<T>) -> Self {
        match value {
            CubeOption::Some(val) => CubeOptionExpand::Some(val.into()),
            CubeOption::None => CubeOptionExpand::None,
        }
    }
}

// Manually implement CubeLaunch as the macro is currently not permissive enough.

pub enum CubeOptionArgs<'a, T: CubeLaunch, R: Runtime> {
    Some(<T as LaunchArg>::RuntimeArg<'a, R>),
    None,
}

impl<'a, T: CubeLaunch, R: Runtime> From<Option<<T as LaunchArg>::RuntimeArg<'a, R>>>
    for CubeOptionArgs<'a, T, R>
{
    fn from(value: Option<<T as LaunchArg>::RuntimeArg<'a, R>>) -> Self {
        match value {
            Some(arg) => Self::Some(arg),
            None => Self::None,
        }
    }
}

impl<T: CubeLaunch, R: Runtime> ArgSettings<R> for CubeOptionArgs<'_, T, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        match self {
            CubeOptionArgs::Some(arg) => {
                arg.register(launcher);
            }
            CubeOptionArgs::None => {}
        }
    }
}
impl<T: CubeLaunch> LaunchArg for CubeOption<T> {
    type RuntimeArg<'a, R: Runtime> = CubeOptionArgs<'a, T, R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            CubeOptionArgs::Some(arg) => {
                CubeOptionCompilationArg::Some(T::compilation_arg::<R>(arg))
            }
            CubeOptionArgs::None => CubeOptionCompilationArg::None,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub enum CubeOptionCompilationArg<T: CubeLaunch> {
    Some(<T as LaunchArgExpand>::CompilationArg),
    None,
}

impl<T: CubeLaunch> Clone for CubeOptionCompilationArg<T> {
    fn clone(&self) -> Self {
        match self {
            CubeOptionCompilationArg::Some(arg) => CubeOptionCompilationArg::Some(arg.clone()),
            CubeOptionCompilationArg::None => CubeOptionCompilationArg::None,
        }
    }
}

impl<T: CubeLaunch> PartialEq for CubeOptionCompilationArg<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CubeOptionCompilationArg::Some(arg_0), CubeOptionCompilationArg::Some(arg_1)) => {
                arg_0 == arg_1
            }
            (CubeOptionCompilationArg::None, CubeOptionCompilationArg::None) => true,
            _ => false,
        }
    }
}

impl<T: CubeLaunch> Eq for CubeOptionCompilationArg<T> {}

impl<T: CubeLaunch> core::hash::Hash for CubeOptionCompilationArg<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            CubeOptionCompilationArg::Some(arg) => {
                arg.hash(state);
            }
            CubeOptionCompilationArg::None => {}
        };
    }
}

impl<T: CubeLaunch> core::fmt::Debug for CubeOptionCompilationArg<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CubeOptionCompilationArg::Some(arg) => f
                .debug_tuple("CubeOptionCompilationArg :: Some")
                .field(arg)
                .finish(),
            CubeOptionCompilationArg::None => write!(f, "CubeOptionCompilationArg :: None"),
        }
    }
}

impl<T: CubeLaunch> CompilationArg for CubeOptionCompilationArg<T> {}

impl<T: CubeLaunch> LaunchArgExpand for CubeOption<T> {
    type CompilationArg = CubeOptionCompilationArg<T>;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            CubeOptionCompilationArg::Some(arg) => CubeOptionExpand::Some(T::expand(arg, builder)),
            CubeOptionCompilationArg::None => CubeOptionExpand::None,
        }
    }

    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            CubeOptionCompilationArg::Some(arg) => {
                CubeOptionExpand::Some(T::expand_output(arg, builder))
            }
            CubeOptionCompilationArg::None => CubeOptionExpand::None,
        }
    }
}
