use crate::{self as cubecl, ExpandType};
use cubecl::prelude::*;

#[doc(hidden)]
#[derive(Default)]
pub enum OptionExpand<T: CubeType> {
    Some(<T as CubeType>::ExpandType),
    #[default]
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

/// Extensions for [`Option`]
#[allow(non_snake_case)]
pub trait CubeOption<T: CubeType> {
    /// Create a new [`Option::Some`] in a kernel
    fn new_Some(_0: T) -> Option<T> {
        Option::Some(_0)
    }
    /// Create a new [`Option::None`] in a kernel
    fn new_None() -> Option<T> {
        Option::None
    }
    #[doc(hidden)]
    fn __expand_Some(_scope: &mut Scope, _0: ExpandType<T>) -> OptionExpand<T> {
        OptionExpand::Some(_0)
    }
    #[doc(hidden)]
    fn __expand_new_Some(_scope: &mut Scope, _0: ExpandType<T>) -> OptionExpand<T> {
        OptionExpand::Some(_0)
    }
    #[doc(hidden)]
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

mod impls {
    use core::ops::{Deref, DerefMut};

    use super::*;
    use OptionExpand::{None, Some};

    #[doc(hidden)]
    impl<T: CubeType> OptionExpand<T> {
        pub fn __expand_is_some_method(&self, _scope: &mut Scope) -> bool {
            matches!(*self, Some(_))
        }

        pub fn __expand_is_some_and_method(
            self,
            scope: &mut Scope,
            f: impl FnOnce(&mut Scope, T::ExpandType) -> bool,
        ) -> bool {
            match self {
                None => false,
                Some(x) => f(scope, x),
            }
        }

        pub fn __expand_is_none_method(&self, _scope: &mut Scope) -> bool {
            !self.is_some()
        }

        pub fn __expand_is_none_or_method(
            self,
            scope: &mut Scope,
            f: impl FnOnce(&mut Scope, T::ExpandType) -> bool,
        ) -> bool {
            match self {
                None => true,
                Some(x) => f(scope, x),
            }
        }

        pub fn __expand_as_ref_method(self, _scope: &mut Scope) -> Self {
            self
        }

        pub fn as_mut(self, _scope: &mut Scope) -> Self {
            self
        }

        fn __expand_len_method(&self, _scope: &mut Scope) -> usize {
            match self {
                Some(_) => 1,
                None => 0,
            }
        }

        pub fn __expand_expect_method(self, _scope: &mut Scope, msg: &str) -> T::ExpandType {
            match self {
                Some(val) => val,
                None => Option::None.expect(msg),
            }
        }

        pub fn __expand_unwrap_method(self, _scope: &mut Scope) -> T::ExpandType {
            match self {
                Some(val) => val,
                None => Option::None.unwrap(),
            }
        }

        pub fn __expand_unwrap_or_method(
            self,
            _scope: &mut Scope,
            default: T::ExpandType,
        ) -> T::ExpandType {
            match self {
                Some(x) => x,
                None => default,
            }
        }

        pub fn __expand_unwrap_or_else_method<F>(self, scope: &mut Scope, f: F) -> T::ExpandType
        where
            F: FnOnce(&mut Scope) -> T::ExpandType,
        {
            match self {
                Some(x) => x,
                None => f(scope),
            }
        }

        pub fn __expand_unwrap_or_default_method(self, _scope: &mut Scope) -> T::ExpandType
        where
            T: Default + Into<T::ExpandType>,
        {
            match self {
                Some(x) => x,
                None => T::default().into(),
            }
        }

        pub fn __expand_map_method<U, F>(self, scope: &mut Scope, f: F) -> OptionExpand<U>
        where
            U: CubeType,
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
        {
            match self {
                Some(x) => Some(f(scope, x)),
                None => None,
            }
        }

        pub fn __expand_inspect_method<F>(self, scope: &mut Scope, f: F) -> Self
        where
            F: FnOnce(&mut Scope, T::ExpandType),
        {
            if let Some(x) = self.clone() {
                f(scope, x);
            }

            self
        }

        pub fn __expand_map_or_method<U, F>(
            self,
            scope: &mut Scope,
            default: U::ExpandType,
            f: F,
        ) -> U::ExpandType
        where
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
            U: CubeType,
        {
            match self {
                Some(t) => f(scope, t),
                None => default,
            }
        }

        pub fn __expand_map_or_else_method<U, D, F>(
            self,
            scope: &mut Scope,
            default: D,
            f: F,
        ) -> U::ExpandType
        where
            U: CubeType,
            D: FnOnce(&mut Scope) -> U::ExpandType,
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
        {
            match self {
                Some(t) => f(scope, t),
                None => default(scope),
            }
        }

        pub fn __expand_map_or_default_method<U, F>(self, scope: &mut Scope, f: F) -> U::ExpandType
        where
            U: CubeType + Default + Into<U::ExpandType>,
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
        {
            match self {
                Some(t) => f(scope, t),
                None => U::default().into(),
            }
        }

        pub fn __expand_as_deref_method(self, scope: &mut Scope) -> OptionExpand<T::Target>
        where
            T: Deref<Target: CubeType + Sized>,
            T::ExpandType: Deref<Target = <T::Target as CubeType>::ExpandType>,
        {
            self.__expand_map_method(scope, |_, it| (*it).clone())
        }

        pub fn __expand_as_deref_mut_method(self, scope: &mut Scope) -> OptionExpand<T::Target>
        where
            T: DerefMut<Target: CubeType + Sized>,
            T::ExpandType: Deref<Target = <T::Target as CubeType>::ExpandType>,
        {
            self.__expand_map_method(scope, |_, it| (*it).clone())
        }

        pub fn __expand_and_method<U>(
            self,
            _scope: &mut Scope,
            optb: OptionExpand<U>,
        ) -> OptionExpand<U>
        where
            U: CubeType,
        {
            match self {
                Some(_) => optb,
                None => None,
            }
        }

        pub fn __expand_and_then_method<U, F>(self, scope: &mut Scope, f: F) -> OptionExpand<U>
        where
            U: CubeType,
            F: FnOnce(&mut Scope, T::ExpandType) -> OptionExpand<U>,
        {
            match self {
                Some(x) => f(scope, x),
                None => None,
            }
        }

        pub fn __expand_filter_method<P>(self, scope: &mut Scope, predicate: P) -> Self
        where
            P: FnOnce(&mut Scope, T::ExpandType) -> bool,
        {
            if let Some(x) = self {
                if predicate(scope, x.clone()) {
                    return Some(x);
                }
            }
            None
        }

        pub fn __expand_or_method(
            self,
            _scope: &mut Scope,
            optb: OptionExpand<T>,
        ) -> OptionExpand<T> {
            match self {
                x @ Some(_) => x,
                None => optb,
            }
        }

        pub fn __expand_or_else_method<F>(self, scope: &mut Scope, f: F) -> OptionExpand<T>
        where
            F: FnOnce(&mut Scope) -> OptionExpand<T>,
        {
            match self {
                x @ Some(_) => x,
                None => f(scope),
            }
        }

        pub fn __expand_xor_method(
            self,
            _scope: &mut Scope,
            optb: OptionExpand<T>,
        ) -> OptionExpand<T> {
            match (self, optb) {
                (a @ Some(_), None) => a,
                (None, b @ Some(_)) => b,
                _ => None,
            }
        }

        // Entry methods that return &mut T excluded for now

        pub fn __expand_take_method(&mut self, _scope: &mut Scope) -> OptionExpand<T> {
            core::mem::take(self)
        }

        pub fn __expand_take_if_method<P>(
            &mut self,
            scope: &mut Scope,
            predicate: P,
        ) -> OptionExpand<T>
        where
            P: FnOnce(&mut Scope, T::ExpandType) -> bool,
        {
            match self {
                Some(value) if predicate(scope, value.clone()) => self.__expand_take_method(scope),
                _ => None,
            }
        }

        pub fn __expand_replace_method(
            &mut self,
            _scope: &mut Scope,
            value: T::ExpandType,
        ) -> OptionExpand<T> {
            core::mem::replace(self, Some(value))
        }

        pub fn __expand_zip_method<U>(
            self,
            _scope: &mut Scope,
            other: OptionExpand<U>,
        ) -> OptionExpand<(T, U)>
        where
            U: CubeType,
        {
            match (self, other) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            }
        }

        pub fn __expand_zip_with_method<U, F, R>(
            self,
            scope: &mut Scope,
            other: OptionExpand<U>,
            f: F,
        ) -> OptionExpand<R>
        where
            F: FnOnce(&mut Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
            R: CubeType,
            U: CubeType,
        {
            match (self, other) {
                (Some(a), Some(b)) => Some(f(scope, a, b)),
                _ => None,
            }
        }

        pub fn __expand_reduce_method<U, R, F>(
            self,
            scope: &mut Scope,
            other: OptionExpand<U>,
            f: F,
        ) -> OptionExpand<R>
        where
            U: CubeType,
            R: CubeType,
            T::ExpandType: Into<R::ExpandType>,
            U::ExpandType: Into<R::ExpandType>,
            F: FnOnce(&mut Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
        {
            match (self, other) {
                (Some(a), Some(b)) => Some(f(scope, a, b)),
                (Some(a), _) => Some(a.into()),
                (_, Some(b)) => Some(b.into()),
                _ => None,
            }
        }
    }

    #[doc(hidden)]
    impl<T: CubeType, U: CubeType> OptionExpand<(T, U)> {
        pub fn __expand_unzip_method(
            self,
            _scope: &mut Scope,
        ) -> (OptionExpand<T>, OptionExpand<U>) {
            match self {
                Some((a, b)) => (Some(a), Some(b)),
                None => (None, None),
            }
        }
    }
}
