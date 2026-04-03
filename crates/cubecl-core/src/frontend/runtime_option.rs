use cubecl_macros::derive_expand;

use crate as cubecl;
use crate::prelude::*;

#[derive_expand(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants, no_constructors)]
pub enum Option<T: CubeType> {
    /// No value.
    None,
    /// Some value of type `T`.
    Some(T),
}

fn discriminant(variant_name: &'static str) -> i32 {
    OptionExpand::<u32>::discriminant_of(variant_name)
}

pub enum OptionArgs<T: LaunchArg, R: Runtime> {
    Some(<T as LaunchArg>::RuntimeArg<R>),
    None,
}

impl<T: LaunchArg, R: Runtime> From<Option<<T as LaunchArg>::RuntimeArg<R>>> for OptionArgs<T, R> {
    fn from(value: Option<<T as LaunchArg>::RuntimeArg<R>>) -> Self {
        match value {
            Some(arg) => Self::Some(arg),
            None => Self::None,
        }
    }
}

impl<T: LaunchArg + Default + IntoRuntime> LaunchArg for Option<T> {
    type RuntimeArg<R: Runtime> = OptionArgs<T, R>;
    type CompilationArg = OptionCompilationArg<T>;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        match arg {
            OptionArgs::Some(arg) => OptionCompilationArg::Some(T::register(arg, launcher)),
            OptionArgs::None => OptionCompilationArg::None,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            OptionCompilationArg::Some(value) => {
                let value = T::expand(value, builder);
                OptionExpand {
                    discriminant: discriminant("Some").into(),
                    value,
                }
            }
            OptionCompilationArg::None => OptionExpand {
                discriminant: discriminant("None").into(),
                value: T::default().__expand_runtime_method(&mut builder.scope),
            },
        }
    }

    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            OptionCompilationArg::Some(value) => {
                let value = T::expand_output(value, builder);
                OptionExpand {
                    discriminant: discriminant("Some").into(),
                    value,
                }
            }
            OptionCompilationArg::None => OptionExpand {
                discriminant: discriminant("None").into(),
                value: T::default().__expand_runtime_method(&mut builder.scope),
            },
        }
    }
}

pub enum OptionCompilationArg<T: LaunchArg> {
    Some(T::CompilationArg),
    None,
}

impl<T: LaunchArg> Clone for OptionCompilationArg<T> {
    fn clone(&self) -> Self {
        match self {
            OptionCompilationArg::Some(value) => OptionCompilationArg::Some(value.clone()),
            OptionCompilationArg::None => OptionCompilationArg::None,
        }
    }
}

impl<T: LaunchArg> PartialEq for OptionCompilationArg<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Some(l0), Self::Some(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<T: LaunchArg> Eq for OptionCompilationArg<T> {}

impl<T: LaunchArg> core::hash::Hash for OptionCompilationArg<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            OptionCompilationArg::Some(value) => value.hash(state),
            OptionCompilationArg::None => {}
        }
    }
}

impl<T: LaunchArg> core::fmt::Debug for OptionCompilationArg<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Some(arg0) => f.debug_tuple("Some").field(arg0).finish(),
            Self::None => write!(f, "None"),
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
    fn none_with_default(_0: T) -> Option<T> {
        Option::None
    }

    #[doc(hidden)]
    fn __expand_Some(scope: &mut Scope, value: T::ExpandType) -> OptionExpand<T> {
        Self::__expand_new_Some(scope, value)
    }
    #[doc(hidden)]
    fn __expand_new_Some(_scope: &mut Scope, value: T::ExpandType) -> OptionExpand<T> {
        OptionExpand::<T> {
            discriminant: discriminant("Some").into(),
            value,
        }
    }
    fn __expand_none_with_default(_scope: &mut Scope, value: T::ExpandType) -> OptionExpand<T> {
        OptionExpand {
            discriminant: discriminant("None").into(),
            value,
        }
    }
}

/// Extensions for [`Option`] that require default
#[allow(non_snake_case)]
pub trait CubeOptionDefault<T: CubeType + Default + IntoRuntime>: CubeOption<T> {
    /// Create a new [`Option::None`] in a kernel
    fn new_None() -> Option<T> {
        Option::None
    }

    #[doc(hidden)]
    fn __expand_new_None(scope: &mut Scope) -> OptionExpand<T> {
        let value = T::default().__expand_runtime_method(scope);
        Self::__expand_none_with_default(scope, value)
    }
}

impl<T: CubeType> CubeOption<T> for Option<T> {}
impl<T: CubeType + Default + IntoRuntime> CubeOptionDefault<T> for Option<T> {}

mod impls {
    use core::ops::{Deref, DerefMut};

    use super::*;
    use crate as cubecl;

    /////////////////////////////////////////////////////////////////////////////
    // Type implementation
    /////////////////////////////////////////////////////////////////////////////

    #[doc(hidden)]
    impl<T: CubeType> OptionExpand<T> {
        pub fn __expand_is_some_and_method(
            self,
            scope: &mut Scope,
            f: impl FnOnce(&mut Scope, T::ExpandType) -> NativeExpand<bool>,
        ) -> NativeExpand<bool> {
            match_expand_expr(scope, self, discriminant("None"), |_, _| false.into())
                .case(scope, discriminant("Some"), |scope, value| f(scope, value))
                .finish(scope)
        }

        pub fn __expand_is_none_or_method(
            self,
            scope: &mut Scope,
            f: impl FnOnce(&mut Scope, T::ExpandType) -> NativeExpand<bool>,
        ) -> NativeExpand<bool> {
            match_expand_expr(scope, self, discriminant("None"), |_, _| true.into())
                .case(scope, discriminant("Some"), |scope, value| f(scope, value))
                .finish(scope)
        }

        pub fn __expand_as_ref_method(&self, _scope: &mut Scope) -> OptionExpand<T> {
            self.clone()
        }

        pub fn __expand_as_mut_method(&mut self, _scope: &mut Scope) -> OptionExpand<T> {
            self.clone()
        }

        pub fn __expand_expect_method(self, scope: &mut Scope, msg: &str) -> T::ExpandType
        where
            T::ExpandType: Assign,
        {
            // Replace with `trap` eventually to ensure execution doesn't continue to the next kernel
            match_expand_expr(scope, self, discriminant("Some"), |_, value| value)
                .case(scope, discriminant("None"), |scope, value| {
                    printf_expand(scope, msg, alloc::vec![]);
                    terminate!();
                    value
                })
                .finish(scope)
        }

        pub fn __expand_unwrap_or_else_method<F>(self, scope: &mut Scope, f: F) -> T::ExpandType
        where
            F: FnOnce(&mut Scope) -> T::ExpandType,
            T::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), |_, value| value)
                .case(scope, discriminant("None"), |scope, _| f(scope))
                .finish(scope)
        }

        pub fn __expand_map_method<U, F>(self, scope: &mut Scope, f: F) -> OptionExpand<U>
        where
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
            U: CubeType + IntoRuntime + Default,
            OptionExpand<U>: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), |scope, value| {
                let value = f(scope, value);
                Option::__expand_new_Some(scope, value)
            })
            .case(scope, discriminant("None"), |scope, _| {
                Option::__expand_new_None(scope)
            })
            .finish(scope)
        }

        pub fn __expand_inspect_method<F>(self, scope: &mut Scope, f: F) -> Self
        where
            F: FnOnce(&mut Scope, &T::ExpandType),
        {
            match_expand(scope, self.clone(), discriminant("Some"), |scope, value| {
                f(scope, &value)
            })
            .case(scope, discriminant("None"), |_, _| {})
            .finish(scope);
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
            U: CubeType + Default + IntoRuntime,
            U::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), f)
                .case(scope, discriminant("None"), |_, _| default)
                .finish(scope)
        }

        pub fn __expand_map_or_else_method<U, D, F>(
            self,
            scope: &mut Scope,
            default: D,
            f: F,
        ) -> U::ExpandType
        where
            D: FnOnce(&mut Scope) -> U::ExpandType,
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
            U: CubeType + Default + IntoRuntime,
            U::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), f)
                .case(scope, discriminant("None"), |scope, _| default(scope))
                .finish(scope)
        }

        pub fn __expand_map_or_default_method<U, F>(self, scope: &mut Scope, f: F) -> U::ExpandType
        where
            U: CubeType + IntoRuntime + Default,
            F: FnOnce(&mut Scope, T::ExpandType) -> U::ExpandType,
            U::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), f)
                .case(scope, discriminant("None"), |scope, _| {
                    U::default().__expand_runtime_method(scope)
                })
                .finish(scope)
        }

        pub fn __expand_as_deref_method(self, scope: &mut Scope) -> OptionExpand<T::Target>
        where
            T: Deref<Target: CubeType + Default + IntoRuntime>,
            T::ExpandType: Deref<Target = <T::Target as CubeType>::ExpandType>,
            <T::Target as CubeType>::ExpandType: Assign,
        {
            self.__expand_map_method(scope, |_, value| (*value).clone())
        }

        pub fn __expand_as_deref_mut_method(self, scope: &mut Scope) -> OptionExpand<T::Target>
        where
            T: DerefMut<Target: CubeType + Default + IntoRuntime>,
            T::ExpandType: Deref<Target = <T::Target as CubeType>::ExpandType>,
            <T::Target as CubeType>::ExpandType: Assign,
        {
            self.__expand_map_method(scope, |_, value| (*value).clone())
        }

        pub fn __expand_and_then_method<U, F>(self, scope: &mut Scope, f: F) -> OptionExpand<U>
        where
            F: FnOnce(&mut Scope, T::ExpandType) -> OptionExpand<U>,
            U: CubeType + IntoRuntime + Default,
            U::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), f)
                .case(scope, discriminant("None"), |scope, _| {
                    Option::__expand_new_None(scope)
                })
                .finish(scope)
        }

        pub fn __expand_filter_method<P>(self, scope: &mut Scope, predicate: P) -> Self
        where
            P: FnOnce(&mut Scope, T::ExpandType) -> NativeExpand<bool>,
            T: Default + IntoRuntime,
            Self: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), |scope, value| {
                let cond = predicate(scope, value.clone());
                if_else_expr_expand(scope, cond, |scope| Option::__expand_new_Some(scope, value))
                    .or_else(scope, |scope| Option::__expand_new_None(scope))
            })
            .case(scope, discriminant("None"), |scope, _| {
                Option::__expand_new_None(scope)
            })
            .finish(scope)
        }

        pub fn __expand_or_else_method<F>(self, scope: &mut Scope, f: F) -> OptionExpand<T>
        where
            F: FnOnce(&mut Scope) -> OptionExpand<T>,
            OptionExpand<T>: Assign,
        {
            let is_some = self.clone().__expand_is_some_method(scope);
            if_else_expr_expand(scope, is_some, |_| self).or_else(scope, |scope| f(scope))
        }

        pub fn __expand_zip_with_method<U, F, R>(
            self,
            scope: &mut Scope,
            other: OptionExpand<U>,
            f: F,
        ) -> OptionExpand<R>
        where
            F: FnOnce(&mut Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
            U: CubeType,
            R: CubeType + IntoRuntime + Default,
            OptionExpand<R>: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), |scope, value| {
                match_expand_expr(scope, other, discriminant("Some"), |scope, other| {
                    let value = f(scope, value, other);
                    Option::__expand_new_Some(scope, value)
                })
                .case(scope, discriminant("None"), |scope, _| {
                    Option::__expand_new_None(scope)
                })
                .finish(scope)
            })
            .case(scope, discriminant("None"), |scope, _| {
                Option::__expand_new_None(scope)
            })
            .finish(scope)
        }

        pub fn __expand_reduce_method<U, R, F>(
            self,
            scope: &mut Scope,
            other: OptionExpand<U>,
            f: F,
        ) -> OptionExpand<R>
        where
            T::ExpandType: Into<R::ExpandType>,
            U::ExpandType: Into<R::ExpandType>,
            F: FnOnce(&mut Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
            U: CubeType + IntoRuntime + Default,
            R: CubeType + IntoRuntime + Default,
            OptionExpand<R>: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), {
                let other = other.clone();
                |scope, value| {
                    match_expand_expr(scope, other, discriminant("Some"), {
                        let value = value.clone();
                        |scope, other| {
                            let value = f(scope, value, other);
                            Option::__expand_new_Some(scope, value)
                        }
                    })
                    .case(scope, discriminant("None"), |scope, _| {
                        Option::__expand_new_Some(scope, value.into())
                    })
                    .finish(scope)
                }
            })
            .case(scope, discriminant("None"), |scope, _| {
                match_expand_expr(scope, other, discriminant("Some"), |scope, other| {
                    Option::__expand_new_Some(scope, other.into())
                })
                .case(scope, discriminant("None"), |scope, _| {
                    Option::__expand_new_None(scope)
                })
                .finish(scope)
            })
            .finish(scope)
        }

        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn __expand_unwrap_unchecked_method(self, scope: &mut Scope) -> T::ExpandType
        where
            T::ExpandType: Assign,
        {
            match_expand_expr(scope, self, discriminant("Some"), |_, value| value).finish(scope)
        }
    }

    #[cube(expand_only)]
    impl<T: CubeType> Option<T> {
        /////////////////////////////////////////////////////////////////////////
        // Querying the contained values
        /////////////////////////////////////////////////////////////////////////

        /// Returns `true` if the option is a [`Some`] value.
        ///
        /// # Examples
        ///
        /// ```
        /// let x: Option<u32> = Some(2);
        /// assert_eq!(x.is_some(), true);
        ///
        /// let x: Option<u32> = None;
        /// assert_eq!(x.is_some(), false);
        /// ```
        pub fn is_some(&self) -> bool {
            match self {
                Option::Some(_) => true.runtime(),
                Option::None => false.runtime(),
            }
        }

        /// Returns `true` if the option is a [`None`] value.
        ///
        /// # Examples
        ///
        /// ```
        /// let x: Option<u32> = Some(2);
        /// assert_eq!(x.is_none(), false);
        ///
        /// let x: Option<u32> = None;
        /// assert_eq!(x.is_none(), true);
        /// ```
        #[must_use = "if you intended to assert that this doesn't have a value, consider \
                  wrapping this in an `assert!()` instead"]
        pub fn is_none(&self) -> bool {
            !self.is_some()
        }

        /////////////////////////////////////////////////////////////////////////
        // Getting to contained values
        /////////////////////////////////////////////////////////////////////////

        /// Returns the contained [`Some`] value, consuming the `self` value.
        ///
        /// Because this function may panic, its use is generally discouraged.
        /// Panics are meant for unrecoverable errors, and
        /// [may abort the entire program][panic-abort].
        ///
        /// Instead, prefer to use pattern matching and handle the [`None`]
        /// case explicitly, or call [`unwrap_or`], [`unwrap_or_else`], or
        /// [`unwrap_or_default`]. In functions returning `Option`, you can use
        /// [the `?` (try) operator][try-option].
        ///
        /// [panic-abort]: https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html
        /// [try-option]: https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#where-the--operator-can-be-used
        /// [`unwrap_or`]: Option::unwrap_or
        /// [`unwrap_or_else`]: Option::unwrap_or_else
        /// [`unwrap_or_default`]: Option::unwrap_or_default
        ///
        /// # Panics
        ///
        /// Panics if the self value equals [`None`].
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some("air");
        /// assert_eq!(x.unwrap(), "air");
        /// ```
        ///
        /// ```should_panic
        /// let x: Option<&str> = None;
        /// assert_eq!(x.unwrap(), "air"); // fails
        /// ```
        pub fn unwrap(self) -> T
        where
            T::ExpandType: Assign,
        {
            self.expect("called `Option::unwrap()` on a `None` value")
        }

        /// Returns the contained [`Some`] value or a provided default.
        ///
        /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
        /// the result of a function call, it is recommended to use [`unwrap_or_else`],
        /// which is lazily evaluated.
        ///
        /// [`unwrap_or_else`]: Option::unwrap_or_else
        ///
        /// # Examples
        ///
        /// ```
        /// assert_eq!(Some("car").unwrap_or("bike"), "car");
        /// assert_eq!(None.unwrap_or("bike"), "bike");
        /// ```
        pub fn unwrap_or(self, default: T) -> T
        where
            T::ExpandType: Assign,
        {
            match self {
                Some(x) => x,
                None => default,
            }
        }

        /// Returns the contained [`Some`] value or a default.
        ///
        /// Consumes the `self` argument then, if [`Some`], returns the contained
        /// value, otherwise if [`None`], returns the [default value] for that
        /// type.
        ///
        /// # Examples
        ///
        /// ```
        /// let x: Option<u32> = None;
        /// let y: Option<u32> = Some(12);
        ///
        /// assert_eq!(x.unwrap_or_default(), 0);
        /// assert_eq!(y.unwrap_or_default(), 12);
        /// ```
        ///
        /// [default value]: Default::default
        /// [`parse`]: str::parse
        /// [`FromStr`]: crate::str::FromStr
        pub fn unwrap_or_default(self) -> T
        where
            T: Default + IntoRuntime,
            T::ExpandType: Assign,
        {
            match self {
                Some(x) => x,
                None => comptime![T::default()].runtime(),
            }
        }

        /////////////////////////////////////////////////////////////////////////
        // Transforming contained values
        /////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////
        // Boolean operations on the values, eager and lazy
        /////////////////////////////////////////////////////////////////////////

        /// Returns [`None`] if the option is [`None`], otherwise returns `optb`.
        ///
        /// Arguments passed to `and` are eagerly evaluated; if you are passing the
        /// result of a function call, it is recommended to use [`and_then`], which is
        /// lazily evaluated.
        ///
        /// [`and_then`]: Option::and_then
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some(2);
        /// let y: Option<&str> = None;
        /// assert_eq!(x.and(y), None);
        ///
        /// let x: Option<u32> = None;
        /// let y = Some("foo");
        /// assert_eq!(x.and(y), None);
        ///
        /// let x = Some(2);
        /// let y = Some("foo");
        /// assert_eq!(x.and(y), Some("foo"));
        ///
        /// let x: Option<u32> = None;
        /// let y: Option<&str> = None;
        /// assert_eq!(x.and(y), None);
        /// ```
        pub fn and<U>(self, optb: Option<U>) -> Option<U>
        where
            U: CubeType + IntoRuntime + Default,
            U::ExpandType: Assign,
        {
            match self {
                Option::Some(_) => optb,
                Option::None => Option::new_None(),
            }
        }

        /// Returns the option if it contains a value, otherwise returns `optb`.
        ///
        /// Arguments passed to `or` are eagerly evaluated; if you are passing the
        /// result of a function call, it is recommended to use [`or_else`], which is
        /// lazily evaluated.
        ///
        /// [`or_else`]: Option::or_else
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some(2);
        /// let y = None;
        /// assert_eq!(x.or(y), Some(2));
        ///
        /// let x = None;
        /// let y = Some(100);
        /// assert_eq!(x.or(y), Some(100));
        ///
        /// let x = Some(2);
        /// let y = Some(100);
        /// assert_eq!(x.or(y), Some(2));
        ///
        /// let x: Option<u32> = None;
        /// let y = None;
        /// assert_eq!(x.or(y), None);
        /// ```
        pub fn or(self, optb: Option<T>) -> Option<T>
        where
            T::ExpandType: Assign,
        {
            if self.is_some() { self } else { optb }
        }

        /// Returns [`Some`] if exactly one of `self`, `optb` is [`Some`], otherwise returns [`None`].
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some(2);
        /// let y: Option<u32> = None;
        /// assert_eq!(x.xor(y), Some(2));
        ///
        /// let x: Option<u32> = None;
        /// let y = Some(2);
        /// assert_eq!(x.xor(y), Some(2));
        ///
        /// let x = Some(2);
        /// let y = Some(2);
        /// assert_eq!(x.xor(y), None);
        ///
        /// let x: Option<u32> = None;
        /// let y: Option<u32> = None;
        /// assert_eq!(x.xor(y), None);
        /// ```
        pub fn xor(self, optb: Option<T>) -> Option<T>
        where
            T: Default + IntoRuntime,
            T::ExpandType: Assign,
        {
            if self.is_some() && optb.is_none() {
                self
            } else if self.is_none() && optb.is_some() {
                optb
            } else {
                Option::new_None()
            }
        }

        /////////////////////////////////////////////////////////////////////////
        // Misc
        /////////////////////////////////////////////////////////////////////////

        // TODO: `take`/`take_if`/`replace`

        /// Zips `self` with another `Option`.
        ///
        /// If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some((s, o))`.
        /// Otherwise, `None` is returned.
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some(1);
        /// let y = Some("hi");
        /// let z = None::<u8>;
        ///
        /// assert_eq!(x.zip(y), Some((1, "hi")));
        /// assert_eq!(x.zip(z), None);
        /// ```
        pub fn zip<U>(self, other: Option<U>) -> Option<(T, U)>
        where
            U: CubeType,
            (T, U): Default + IntoRuntime,
            (T::ExpandType, U::ExpandType): Into<<(T, U) as CubeType>::ExpandType>,
            OptionExpand<(T, U)>: Assign,
        {
            match self {
                Some(a) => match other {
                    Some(b) => Option::Some((a, b)),
                    None => Option::new_None(),
                },
                None => Option::new_None(),
            }
        }
    }

    #[cube(expand_only)]
    impl<
        T: CubeType<ExpandType: Assign> + IntoRuntime + Default,
        U: CubeType<ExpandType: Assign> + IntoRuntime + Default,
    > Option<(T, U)>
    {
        /// Unzips an option containing a tuple of two options.
        ///
        /// If `self` is `Some((a, b))` this method returns `(Some(a), Some(b))`.
        /// Otherwise, `(None, None)` is returned.
        ///
        /// # Examples
        ///
        /// ```
        /// let x = Some((1, "hi"));
        /// let y = None::<(u8, u32)>;
        ///
        /// assert_eq!(x.unzip(), (Some(1), Some("hi")));
        /// assert_eq!(y.unzip(), (None, None));
        /// ```
        #[inline]
        pub fn unzip(self) -> (Option<T>, Option<U>) {
            match self {
                Option::Some(value) => (Option::Some(value.0), Option::Some(value.1)),
                Option::None => (Option::new_None(), Option::new_None()),
            }
        }
    }
}
