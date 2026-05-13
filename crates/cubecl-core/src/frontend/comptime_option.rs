use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_macros::derive_expand;

#[derive(Default, Clone, Copy)]
pub enum ComptimeOption<T> {
    #[default]
    None,
    Some(T),
}

// Separate implementation so we don't need `CubeType` for `ComptimeOption` itself.
// This is important because `&T where T: CubeType` does not necessarily implement
// `CubeType`, but we need to support it in `as_ref`/`as_mut`.
#[derive_expand(CubeType)]
pub enum ComptimeOption<T: CubeType> {
    None,
    Some(T),
}

impl<T: CubeType<ExpandType: Clone>> Clone for ComptimeOptionExpand<T> {
    fn clone(&self) -> Self {
        self.clone_unchecked()
    }
}

impl<T: CubeType<ExpandType: Copy>> Copy for ComptimeOptionExpand<T> {}

#[allow(clippy::derivable_impls)]
impl<T: CubeType> Default for ComptimeOptionExpand<T> {
    fn default() -> Self {
        Self::None
    }
}

#[allow(non_snake_case)]
impl<T: CubeType> ComptimeOption<T> {
    pub fn __expand_Some(scope: &Scope, value: T::ExpandType) -> ComptimeOptionExpand<T> {
        Self::__expand_new_Some(scope, value)
    }
}

impl<T: CubeType> ComptimeOptionExpand<T> {
    pub fn is_some(&self) -> bool {
        match self {
            ComptimeOptionExpand::Some(_) => true,
            ComptimeOptionExpand::None => false,
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
            ComptimeOptionExpand::Some(val) => val,
            ComptimeOptionExpand::None => fallback,
        }
    }
}

pub enum ComptimeOptionArgs<T: LaunchArg, R: Runtime> {
    Some(<T as LaunchArg>::RuntimeArg<R>),
    None,
}

impl<T: LaunchArg, R: Runtime> From<Option<<T as LaunchArg>::RuntimeArg<R>>>
    for ComptimeOptionArgs<T, R>
{
    fn from(value: Option<<T as LaunchArg>::RuntimeArg<R>>) -> Self {
        match value {
            Some(arg) => Self::Some(arg),
            None => Self::None,
        }
    }
}

impl<T: LaunchArg + 'static + CubeType> LaunchArg for ComptimeOption<T> {
    type RuntimeArg<R: Runtime> = ComptimeOptionArgs<T, R>;
    type CompilationArg = ComptimeOptionCompilationArg<T>;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        match arg {
            ComptimeOptionArgs::Some(arg) => {
                ComptimeOptionCompilationArg::Some(T::register(arg, launcher))
            }
            ComptimeOptionArgs::None => ComptimeOptionCompilationArg::None,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        match arg {
            ComptimeOptionCompilationArg::Some(arg) => {
                ComptimeOptionExpand::Some(T::expand(arg, builder))
            }
            ComptimeOptionCompilationArg::None => ComptimeOptionExpand::None,
        }
    }
}

pub enum ComptimeOptionCompilationArg<T: LaunchArg> {
    Some(<T as LaunchArg>::CompilationArg),
    None,
}

impl<T: LaunchArg> Clone for ComptimeOptionCompilationArg<T> {
    fn clone(&self) -> Self {
        match self {
            ComptimeOptionCompilationArg::Some(arg) => {
                ComptimeOptionCompilationArg::Some(arg.clone())
            }
            ComptimeOptionCompilationArg::None => ComptimeOptionCompilationArg::None,
        }
    }
}

impl<T: LaunchArg> PartialEq for ComptimeOptionCompilationArg<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ComptimeOptionCompilationArg::Some(arg_0),
                ComptimeOptionCompilationArg::Some(arg_1),
            ) => arg_0 == arg_1,
            (ComptimeOptionCompilationArg::None, ComptimeOptionCompilationArg::None) => true,
            _ => false,
        }
    }
}

impl<T: LaunchArg> Eq for ComptimeOptionCompilationArg<T> {}

impl<T: LaunchArg> core::hash::Hash for ComptimeOptionCompilationArg<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            ComptimeOptionCompilationArg::Some(arg) => {
                arg.hash(state);
            }
            ComptimeOptionCompilationArg::None => {}
        };
    }
}

impl<T: LaunchArg> core::fmt::Debug for ComptimeOptionCompilationArg<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComptimeOptionCompilationArg::Some(arg) => f.debug_tuple("Some").field(arg).finish(),
            ComptimeOptionCompilationArg::None => write!(f, "None"),
        }
    }
}

mod impls {
    use core::ops::{Deref, DerefMut};

    use super::*;
    use ComptimeOption::Some;
    type Option<T> = ComptimeOption<T>;
    type OptionExpand<T> = ComptimeOptionExpand<T>;

    /////////////////////////////////////////////////////////////////////////////
    // Type implementation
    /////////////////////////////////////////////////////////////////////////////

    mod base {
        use super::*;
        use ComptimeOption::{None, Some};

        impl<T> ComptimeOption<T> {
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
            #[must_use = "if you intended to assert that this has a value, consider `.unwrap()` instead"]
            pub fn is_some(&self) -> bool {
                matches!(*self, Some(_))
            }

            /// Returns `true` if the option is a [`Some`] and the value inside of it matches a predicate.
            ///
            /// # Examples
            ///
            /// ```
            /// let x: Option<u32> = Some(2);
            /// assert_eq!(x.is_some_and(|x| x > 1), true);
            ///
            /// let x: Option<u32> = Some(0);
            /// assert_eq!(x.is_some_and(|x| x > 1), false);
            ///
            /// let x: Option<u32> = None;
            /// assert_eq!(x.is_some_and(|x| x > 1), false);
            ///
            /// let x: Option<String> = Some("ownership".to_string());
            /// assert_eq!(x.as_ref().is_some_and(|x| x.len() > 1), true);
            /// println!("still alive {:?}", x);
            /// ```
            #[must_use]
            pub fn is_some_and(self, f: impl FnOnce(T) -> bool) -> bool {
                match self {
                    None => false,
                    Some(x) => f(x),
                }
            }

            /// Returns `true` if the option is a [`None`] or the value inside of it matches a predicate.
            ///
            /// # Examples
            ///
            /// ```
            /// let x: Option<u32> = Some(2);
            /// assert_eq!(x.is_none_or(|x| x > 1), true);
            ///
            /// let x: Option<u32> = Some(0);
            /// assert_eq!(x.is_none_or(|x| x > 1), false);
            ///
            /// let x: Option<u32> = None;
            /// assert_eq!(x.is_none_or(|x| x > 1), true);
            ///
            /// let x: Option<String> = Some("ownership".to_string());
            /// assert_eq!(x.as_ref().is_none_or(|x| x.len() > 1), true);
            /// println!("still alive {:?}", x);
            /// ```
            #[must_use]
            pub fn is_none_or(self, f: impl FnOnce(T) -> bool) -> bool {
                match self {
                    None => true,
                    Some(x) => f(x),
                }
            }

            /// Converts from `&Option<T>` to `Option<&T>`.
            ///
            /// # Examples
            ///
            /// Calculates the length of an <code>Option<[String]></code> as an <code>Option<[usize]></code>
            /// without moving the [`String`]. The [`map`] method takes the `self` argument by value,
            /// consuming the original, so this technique uses `as_ref` to first take an `Option` to a
            /// reference to the value inside the original.
            ///
            /// [`map`]: Option::map
            /// [String]: ../../std/string/struct.String.html "String"
            /// [`String`]: ../../std/string/struct.String.html "String"
            ///
            /// ```
            /// let text: Option<String> = Some("Hello, world!".to_string());
            /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
            /// // then consume *that* with `map`, leaving `text` on the stack.
            /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
            /// println!("still can print text: {text:?}");
            /// ```
            pub fn as_ref(&self) -> Option<&T> {
                match *self {
                    Some(ref x) => Some(x),
                    None => None,
                }
            }

            /// Converts from `&mut Option<T>` to `Option<&mut T>`.
            ///
            /// # Examples
            ///
            /// ```
            /// let mut x = Some(2);
            /// match x.as_mut() {
            ///     Some(v) => *v = 42,
            ///     None => {},
            /// }
            /// assert_eq!(x, Some(42));
            /// ```
            pub fn as_mut(&mut self) -> Option<&mut T> {
                match *self {
                    Some(ref mut x) => Some(x),
                    None => None,
                }
            }

            /// Returns the contained [`Some`] value, consuming the `self` value.
            ///
            /// # Panics
            ///
            /// Panics if the value is a [`None`] with a custom panic message provided by
            /// `msg`.
            ///
            /// # Examples
            ///
            /// ```
            /// let x = Some("value");
            /// assert_eq!(x.expect("fruits are healthy"), "value");
            /// ```
            ///
            /// ```should_panic
            /// let x: Option<&str> = None;
            /// x.expect("fruits are healthy"); // panics with `fruits are healthy`
            /// ```
            ///
            /// # Recommended Message Style
            ///
            /// We recommend that `expect` messages are used to describe the reason you
            /// _expect_ the `Option` should be `Some`.
            ///
            /// ```should_panic
            /// # let slice: &[u8] = &[];
            /// let item = slice.get(0)
            ///     .expect("slice should not be empty");
            /// ```
            ///
            /// **Hint**: If you're having trouble remembering how to phrase expect
            /// error messages remember to focus on the word "should" as in "env
            /// variable should be set by blah" or "the given binary should be available
            /// and executable by the current user".
            ///
            /// For more detail on expect message styles and the reasoning behind our
            /// recommendation please refer to the section on ["Common Message
            /// Styles"](../../std/error/index.html#common-message-styles) in the [`std::error`](../../std/error/index.html) module docs.
            #[track_caller]
            pub fn expect(self, msg: &str) -> T {
                match self {
                    Some(val) => val,
                    None => panic!("{msg}"),
                }
            }

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
            pub fn unwrap(self) -> T {
                match self {
                    Some(val) => val,
                    None => panic!("called `Option::unwrap()` on a `None` value"),
                }
            }

            /// Returns the contained [`Some`] value or computes it from a closure.
            ///
            /// # Examples
            ///
            /// ```
            /// let k = 10;
            /// assert_eq!(Some(4).unwrap_or_else(|| 2 * k), 4);
            /// assert_eq!(None.unwrap_or_else(|| 2 * k), 20);
            /// ```
            pub fn unwrap_or_else<F>(self, f: F) -> T
            where
                F: FnOnce() -> T,
            {
                match self {
                    Some(x) => x,
                    None => f(),
                }
            }

            /// Maps an `Option<T>` to `Option<U>` by applying a function to a contained value (if `Some`) or returns `None` (if `None`).
            ///
            /// # Examples
            ///
            /// Calculates the length of an <code>Option<[String]></code> as an
            /// <code>Option<[usize]></code>, consuming the original:
            ///
            /// [String]: ../../std/string/struct.String.html "String"
            /// ```
            /// let maybe_some_string = Some(String::from("Hello, World!"));
            /// // `Option::map` takes self *by value*, consuming `maybe_some_string`
            /// let maybe_some_len = maybe_some_string.map(|s| s.len());
            /// assert_eq!(maybe_some_len, Some(13));
            ///
            /// let x: Option<&str> = None;
            /// assert_eq!(x.map(|s| s.len()), None);
            /// ```
            pub fn map<U, F>(self, f: F) -> Option<U>
            where
                F: FnOnce(T) -> U,
            {
                match self {
                    Some(x) => Some(f(x)),
                    None => None,
                }
            }

            /// Calls a function with a reference to the contained value if [`Some`].
            ///
            /// Returns the original option.
            ///
            /// # Examples
            ///
            /// ```
            /// let list = vec![1, 2, 3];
            ///
            /// // prints "got: 2"
            /// let x = list
            ///     .get(1)
            ///     .inspect(|x| println!("got: {x}"))
            ///     .expect("list should be long enough");
            ///
            /// // prints nothing
            /// list.get(5).inspect(|x| println!("got: {x}"));
            /// ```
            pub fn inspect<F>(self, f: F) -> Self
            where
                F: FnOnce(&T),
            {
                if let Some(ref x) = self {
                    f(x);
                }

                self
            }

            /// Returns the provided default result (if none),
            /// or applies a function to the contained value (if any).
            ///
            /// Arguments passed to `map_or` are eagerly evaluated; if you are passing
            /// the result of a function call, it is recommended to use [`map_or_else`],
            /// which is lazily evaluated.
            ///
            /// [`map_or_else`]: Option::map_or_else
            ///
            /// # Examples
            ///
            /// ```
            /// let x = Some("foo");
            /// assert_eq!(x.map_or(42, |v| v.len()), 3);
            ///
            /// let x: Option<&str> = None;
            /// assert_eq!(x.map_or(42, |v| v.len()), 42);
            /// ```
            pub fn map_or<U, F>(self, default: U, f: F) -> U
            where
                F: FnOnce(T) -> U,
            {
                match self {
                    Some(t) => f(t),
                    None => default,
                }
            }
            /// Computes a default function result (if none), or
            /// applies a different function to the contained value (if any).
            ///
            /// # Basic examples
            ///
            /// ```
            /// let k = 21;
            ///
            /// let x = Some("foo");
            /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 3);
            ///
            /// let x: Option<&str> = None;
            /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 42);
            /// ```
            ///
            /// # Handling a Result-based fallback
            ///
            /// A somewhat common occurrence when dealing with optional values
            /// in combination with [`Result<T, E>`] is the case where one wants to invoke
            /// a fallible fallback if the option is not present.  This example
            /// parses a command line argument (if present), or the contents of a file to
            /// an integer.  However, unlike accessing the command line argument, reading
            /// the file is fallible, so it must be wrapped with `Ok`.
            ///
            /// ```no_run
            /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
            /// let v: u64 = std::env::args()
            ///    .nth(1)
            ///    .map_or_else(|| std::fs::read_to_string("/etc/someconfig.conf"), Ok)?
            ///    .parse()?;
            /// #   Ok(())
            /// # }
            /// ```
            pub fn map_or_else<U, D, F>(self, default: D, f: F) -> U
            where
                D: FnOnce() -> U,
                F: FnOnce(T) -> U,
            {
                match self {
                    Some(t) => f(t),
                    None => default(),
                }
            }

            /// Maps an `Option<T>` to a `U` by applying function `f` to the contained
            /// value if the option is [`Some`], otherwise if [`None`], returns the
            /// [default value] for the type `U`.
            ///
            /// # Examples
            ///
            /// ```ignore
            ///
            /// let x: Option<&str> = Some("hi");
            /// let y: Option<&str> = None;
            ///
            /// assert_eq!(x.map_or_default(|x| x.len()), 2);
            /// assert_eq!(y.map_or_default(|y| y.len()), 0);
            /// ```
            ///
            /// [default value]: Default::default
            pub fn map_or_default<U, F>(self, f: F) -> U
            where
                U: Default,
                F: FnOnce(T) -> U,
            {
                match self {
                    Some(t) => f(t),
                    None => U::default(),
                }
            }

            /// Converts from `Option<T>` (or `&Option<T>`) to `Option<&T::Target>`.
            ///
            /// Leaves the original Option in-place, creating a new one with a reference
            /// to the original one, additionally coercing the contents via [`Deref`].
            ///
            /// # Examples
            ///
            /// ```
            /// let x: Option<String> = Some("hey".to_owned());
            /// assert_eq!(x.as_deref(), Some("hey"));
            ///
            /// let x: Option<String> = None;
            /// assert_eq!(x.as_deref(), None);
            /// ```
            pub fn as_deref<'a>(&'a self) -> Option<&'a T::Target>
            where
                T: Deref,
                &'a T: CubeType,
            {
                self.as_ref().map(Deref::deref)
            }

            /// Converts from `Option<T>` (or `&mut Option<T>`) to `Option<&mut T::Target>`.
            ///
            /// Leaves the original `Option` in-place, creating a new one containing a mutable reference to
            /// the inner type's [`Deref::Target`] type.
            ///
            /// # Examples
            ///
            /// ```
            /// let mut x: Option<String> = Some("hey".to_owned());
            /// assert_eq!(x.as_deref_mut().map(|x| {
            ///     x.make_ascii_uppercase();
            ///     x
            /// }), Some("HEY".to_owned().as_mut_str()));
            /// ```
            pub fn as_deref_mut<'a>(&'a mut self) -> Option<&'a mut T::Target>
            where
                T: DerefMut,
                &'a mut T: CubeType,
            {
                self.as_mut().map(DerefMut::deref_mut)
            }

            /// Returns [`None`] if the option is [`None`], otherwise calls `f` with the
            /// wrapped value and returns the result.
            ///
            /// Some languages call this operation flatmap.
            ///
            /// # Examples
            ///
            /// ```
            /// fn sq_then_to_string(x: u32) -> Option<String> {
            ///     x.checked_mul(x).map(|sq| sq.to_string())
            /// }
            ///
            /// assert_eq!(Some(2).and_then(sq_then_to_string), Some(4.to_string()));
            /// assert_eq!(Some(1_000_000).and_then(sq_then_to_string), None); // overflowed!
            /// assert_eq!(None.and_then(sq_then_to_string), None);
            /// ```
            ///
            /// Often used to chain fallible operations that may return [`None`].
            ///
            /// ```
            /// let arr_2d = [["A0", "A1"], ["B0", "B1"]];
            ///
            /// let item_0_1 = arr_2d.get(0).and_then(|row| row.get(1));
            /// assert_eq!(item_0_1, Some(&"A1"));
            ///
            /// let item_2_0 = arr_2d.get(2).and_then(|row| row.get(0));
            /// assert_eq!(item_2_0, None);
            /// ```
            pub fn and_then<U, F>(self, f: F) -> Option<U>
            where
                F: FnOnce(T) -> Option<U>,
                U: CubeType,
            {
                match self {
                    Some(x) => f(x),
                    None => None,
                }
            }

            /// Returns [`None`] if the option is [`None`], otherwise calls `predicate`
            /// with the wrapped value and returns:
            ///
            /// - [`Some(t)`] if `predicate` returns `true` (where `t` is the wrapped
            ///   value), and
            /// - [`None`] if `predicate` returns `false`.
            ///
            /// This function works similar to [`Iterator::filter()`]. You can imagine
            /// the `Option<T>` being an iterator over one or zero elements. `filter()`
            /// lets you decide which elements to keep.
            ///
            /// # Examples
            ///
            /// ```rust
            /// fn is_even(n: &i32) -> bool {
            ///     n % 2 == 0
            /// }
            ///
            /// assert_eq!(None.filter(is_even), None);
            /// assert_eq!(Some(3).filter(is_even), None);
            /// assert_eq!(Some(4).filter(is_even), Some(4));
            /// ```
            ///
            /// [`Some(t)`]: Some
            pub fn filter<P>(self, predicate: P) -> Self
            where
                P: FnOnce(&T) -> bool,
            {
                if let Some(x) = self
                    && predicate(&x)
                {
                    return Some(x);
                }
                None
            }

            /// Returns the option if it contains a value, otherwise calls `f` and
            /// returns the result.
            ///
            /// # Examples
            ///
            /// ```
            /// fn nobody() -> Option<&'static str> { None }
            /// fn vikings() -> Option<&'static str> { Some("vikings") }
            ///
            /// assert_eq!(Some("barbarians").or_else(vikings), Some("barbarians"));
            /// assert_eq!(None.or_else(vikings), Some("vikings"));
            /// assert_eq!(None.or_else(nobody), None);
            /// ```
            pub fn or_else<F>(self, f: F) -> Option<T>
            where
                F: FnOnce() -> Option<T>,
            {
                match self {
                    x @ Some(_) => x,
                    None => f(),
                }
            }

            /// Zips `self` and another `Option` with function `f`.
            ///
            /// If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some(f(s, o))`.
            /// Otherwise, `None` is returned.
            ///
            /// # Examples
            ///
            /// ```ignore
            ///
            /// #[derive(Debug, PartialEq)]
            /// struct Point {
            ///     x: f64,
            ///     y: f64,
            /// }
            ///
            /// impl Point {
            ///     fn new(x: f64, y: f64) -> Self {
            ///         Self { x, y }
            ///     }
            /// }
            ///
            /// let x = Some(17.5);
            /// let y = Some(42.7);
            ///
            /// assert_eq!(x.zip_with(y, Point::new), Some(Point { x: 17.5, y: 42.7 }));
            /// assert_eq!(x.zip_with(None, Point::new), None);
            /// ```
            pub fn zip_with<U, F, R>(self, other: Option<U>, f: F) -> Option<R>
            where
                F: FnOnce(T, U) -> R,
                U: CubeType,
                R: CubeType,
            {
                match (self, other) {
                    (Some(a), Some(b)) => Some(f(a, b)),
                    _ => None,
                }
            }

            /// Reduces two options into one, using the provided function if both are `Some`.
            ///
            /// If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some(f(s, o))`.
            /// Otherwise, if only one of `self` and `other` is `Some`, that one is returned.
            /// If both `self` and `other` are `None`, `None` is returned.
            ///
            /// # Examples
            ///
            /// ```ignore
            ///
            /// let s12 = Some(12);
            /// let s17 = Some(17);
            /// let n = None;
            /// let f = |a, b| a + b;
            ///
            /// assert_eq!(s12.reduce(s17, f), Some(29));
            /// assert_eq!(s12.reduce(n, f), Some(12));
            /// assert_eq!(n.reduce(s17, f), Some(17));
            /// assert_eq!(n.reduce(n, f), None);
            /// ```
            pub fn reduce<U, R, F>(self, other: Option<U>, f: F) -> Option<R>
            where
                T: Into<R>,
                U: Into<R>,
                F: FnOnce(T, U) -> R,
            {
                match (self, other) {
                    (Some(a), Some(b)) => Some(f(a, b)),
                    (Some(a), _) => Some(a.into()),
                    (_, Some(b)) => Some(b.into()),
                    _ => None,
                }
            }
        }

        impl<T> ComptimeOption<T> {
            /////////////////////////////////////////////////////////////////////////
            // Querying the contained values
            /////////////////////////////////////////////////////////////////////////

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
            pub fn unwrap_or(self, default: T) -> T {
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
            {
                match self {
                    Some(x) => x,
                    None => comptime![T::default()].runtime(),
                }
            }

            /// Returns the contained [`Some`] value, consuming the `self` value,
            /// without checking that the value is not [`None`].
            ///
            /// # Safety
            ///
            /// Calling this method on [`None`] is *[undefined behavior]*.
            ///
            /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
            ///
            /// # Examples
            ///
            /// ```
            /// let x = Some("air");
            /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air");
            /// ```
            ///
            /// ```no_run
            /// let x: Option<&str> = None;
            /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air"); // Undefined behavior!
            /// ```
            pub unsafe fn unwrap_unchecked(self) -> T {
                match self {
                    Some(val) => val,
                    // SAFETY: the safety contract must be upheld by the caller.
                    None => comptime![unsafe { core::hint::unreachable_unchecked() }],
                }
            }

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
                U: CubeType,
            {
                match self {
                    Some(_) => optb,
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
            pub fn or(self, optb: Option<T>) -> Option<T> {
                match self {
                    x @ Some(_) => x,
                    None => optb,
                }
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
            pub fn xor(self, optb: Option<T>) -> Option<T> {
                match (self, optb) {
                    (a @ Some(_), None) => a,
                    (None, b @ Some(_)) => b,
                    _ => Option::None,
                }
            }

            /////////////////////////////////////////////////////////////////////////
            // Misc
            /////////////////////////////////////////////////////////////////////////

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
            {
                match (self, other) {
                    (Some(a), Some(b)) => Option::Some((a, b)),
                    _ => Option::None,
                }
            }
        }
    }

    mod expand {
        use super::*;
        use ComptimeOptionExpand::{None, Some};

        #[doc(hidden)]
        impl<T: CubeType> ComptimeOptionExpand<T> {
            pub fn __expand_is_some_method(&self, _scope: &Scope) -> bool {
                matches!(*self, Some(_))
            }

            pub fn __expand_is_some_and_method(
                self,
                scope: &Scope,
                f: impl FnOnce(&Scope, T::ExpandType) -> bool,
            ) -> bool {
                match self {
                    None => false,
                    Some(x) => f(scope, x),
                }
            }

            pub fn __expand_is_none_or_method(
                self,
                scope: &Scope,
                f: impl FnOnce(&Scope, T::ExpandType) -> bool,
            ) -> bool {
                match self {
                    None => true,
                    Some(x) => f(scope, x),
                }
            }

            fn __expand_len_method(&self, _scope: &Scope) -> usize {
                match self {
                    Some(_) => 1,
                    None => 0,
                }
            }

            pub fn __expand_expect_method(self, _scope: &Scope, msg: &str) -> T::ExpandType {
                match self {
                    Some(val) => val,
                    None => panic!("{msg}"),
                }
            }

            #[allow(clippy::unnecessary_literal_unwrap)]
            pub fn __expand_unwrap_method(self, _scope: &Scope) -> T::ExpandType {
                match self {
                    Some(val) => val,
                    None => core::option::Option::None.unwrap(),
                }
            }

            pub fn __expand_unwrap_or_else_method<F>(self, scope: &Scope, f: F) -> T::ExpandType
            where
                F: FnOnce(&Scope) -> T::ExpandType,
            {
                match self {
                    Some(x) => x,
                    None => f(scope),
                }
            }

            pub fn __expand_map_method<U, F>(self, scope: &Scope, f: F) -> ComptimeOptionExpand<U>
            where
                U: CubeType,
                F: FnOnce(&Scope, T::ExpandType) -> U::ExpandType,
            {
                match self {
                    Some(x) => Some(f(scope, x)),
                    None => None,
                }
            }

            pub fn __expand_as_ref_method(&self, _scope: &Scope) -> ComptimeOptionExpand<&T> {
                match self {
                    Some(x) => Some(x),
                    None => None,
                }
            }

            pub fn __expand_as_mut_method(
                &mut self,
                _scope: &Scope,
            ) -> ComptimeOptionExpand<&mut T> {
                match self {
                    Some(x) => Some(x),
                    None => None,
                }
            }

            pub fn __expand_inspect_method<F>(self, scope: &Scope, f: F) -> Self
            where
                F: FnOnce(&Scope, &T::ExpandType),
            {
                if let Some(x) = &self {
                    f(scope, x);
                }

                self
            }

            pub fn __expand_map_or_method<U, F>(
                self,
                scope: &Scope,
                default: U::ExpandType,
                f: F,
            ) -> U::ExpandType
            where
                F: FnOnce(&Scope, T::ExpandType) -> U::ExpandType,
                U: CubeType,
            {
                match self {
                    Some(t) => f(scope, t),
                    None => default,
                }
            }

            pub fn __expand_map_or_else_method<U, D, F>(
                self,
                scope: &Scope,
                default: D,
                f: F,
            ) -> U::ExpandType
            where
                U: CubeType,
                D: FnOnce(&Scope) -> U::ExpandType,
                F: FnOnce(&Scope, T::ExpandType) -> U::ExpandType,
            {
                match self {
                    Some(t) => f(scope, t),
                    None => default(scope),
                }
            }

            pub fn __expand_map_or_default_method<U, F>(self, scope: &Scope, f: F) -> U::ExpandType
            where
                U: CubeType + Default + Into<U::ExpandType>,
                F: FnOnce(&Scope, T::ExpandType) -> U::ExpandType,
            {
                match self {
                    Some(t) => f(scope, t),
                    None => U::default().into(),
                }
            }

            pub fn __expand_as_deref_method(self, scope: &Scope) -> ComptimeOptionExpand<T::Target>
            where
                T: Deref<Target: CubeType + Sized>,
                T::ExpandType: DerefExpand<Target = <T::Target as CubeType>::ExpandType>,
            {
                self.__expand_map_method(scope, |scope, it| it.__expand_deref_method(scope))
            }

            pub fn __expand_as_deref_mut_method(
                self,
                scope: &Scope,
            ) -> ComptimeOptionExpand<T::Target>
            where
                T: DerefMut<Target: CubeType + Sized>,
                T::ExpandType: DerefExpand<Target = <T::Target as CubeType>::ExpandType>,
            {
                self.__expand_map_method(scope, |scope, it| it.__expand_deref_method(scope))
            }

            pub fn __expand_and_then_method<U, F>(
                self,
                scope: &Scope,
                f: F,
            ) -> ComptimeOptionExpand<U>
            where
                U: CubeType,
                F: FnOnce(&Scope, T::ExpandType) -> ComptimeOptionExpand<U>,
            {
                match self {
                    Some(x) => f(scope, x),
                    None => None,
                }
            }

            pub fn __expand_filter_method<P>(self, scope: &Scope, predicate: P) -> Self
            where
                P: FnOnce(&Scope, &T::ExpandType) -> bool,
            {
                if let Some(x) = self
                    && predicate(scope, &x)
                {
                    Some(x)
                } else {
                    None
                }
            }

            pub fn __expand_or_else_method<F>(self, scope: &Scope, f: F) -> ComptimeOptionExpand<T>
            where
                F: FnOnce(&Scope) -> ComptimeOptionExpand<T>,
            {
                match self {
                    x @ Some(_) => x,
                    None => f(scope),
                }
            }

            // Entry methods that return &mut T excluded for now

            pub fn __expand_take_method(&mut self, _scope: &Scope) -> ComptimeOptionExpand<T> {
                core::mem::take(self)
            }

            pub fn __expand_take_if_method<P>(
                &mut self,
                scope: &Scope,
                predicate: P,
            ) -> ComptimeOptionExpand<T>
            where
                P: FnOnce(&Scope, &mut T::ExpandType) -> bool,
            {
                match self {
                    Some(value) => {
                        if predicate(scope, value) {
                            self.__expand_take_method(scope)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }

            pub fn __expand_replace_method(
                &mut self,
                _scope: &Scope,
                value: T::ExpandType,
            ) -> ComptimeOptionExpand<T> {
                core::mem::replace(self, Some(value))
            }

            pub fn __expand_zip_with_method<U, F, R>(
                self,
                scope: &Scope,
                other: ComptimeOptionExpand<U>,
                f: F,
            ) -> ComptimeOptionExpand<R>
            where
                F: FnOnce(&Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
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
                scope: &Scope,
                other: ComptimeOptionExpand<U>,
                f: F,
            ) -> ComptimeOptionExpand<R>
            where
                U: CubeType,
                R: CubeType,
                T::ExpandType: Into<R::ExpandType>,
                U::ExpandType: Into<R::ExpandType>,
                F: FnOnce(&Scope, T::ExpandType, U::ExpandType) -> R::ExpandType,
            {
                match (self, other) {
                    (Some(a), Some(b)) => Some(f(scope, a, b)),
                    (Some(a), _) => Some(a.into()),
                    (_, Some(b)) => Some(b.into()),
                    _ => None,
                }
            }
        }

        impl<T: CubeType> ComptimeOptionExpand<T> {
            pub fn __expand_is_none_method(self, scope: &Scope) -> bool {
                !self.__expand_is_some_method(scope)
            }
            pub fn __expand_unwrap_or_method(
                self,
                _scope: &Scope,
                default: <T as cubecl::prelude::CubeType>::ExpandType,
            ) -> <T as cubecl::prelude::CubeType>::ExpandType {
                {
                    match self {
                        OptionExpand::Some(x) => x,
                        OptionExpand::None => default,
                    }
                }
            }
            pub fn __expand_unwrap_or_default_method(
                self,
                scope: &Scope,
            ) -> <T as cubecl::prelude::CubeType>::ExpandType
            where
                T: Default + IntoRuntime,
            {
                {
                    match self {
                        OptionExpand::Some(x) => x,
                        OptionExpand::None => { T::default() }.__expand_runtime_method(scope),
                    }
                }
            }
            pub fn __expand_unwrap_unchecked_method(
                self,
                _scope: &Scope,
            ) -> <T as cubecl::prelude::CubeType>::ExpandType {
                {
                    match self {
                        OptionExpand::Some(val) => val,
                        OptionExpand::None => unsafe { core::hint::unreachable_unchecked() },
                    }
                }
            }
            pub fn __expand_and_method<U>(
                self,
                scope: &Scope,
                optb: <Option<U> as cubecl::prelude::CubeType>::ExpandType,
            ) -> <Option<U> as cubecl::prelude::CubeType>::ExpandType
            where
                U: CubeType,
            {
                {
                    match self {
                        OptionExpand::Some(_) => optb,
                        OptionExpand::None => Option::__expand_new_None(scope),
                    }
                }
            }
            pub fn __expand_or_method(
                self,
                _scope: &Scope,
                optb: <Option<T> as cubecl::prelude::CubeType>::ExpandType,
            ) -> <Option<T> as cubecl::prelude::CubeType>::ExpandType {
                {
                    match self {
                        x @ OptionExpand::Some(_) => x,
                        OptionExpand::None => optb,
                    }
                }
            }
            pub fn __expand_xor_method(
                self,
                scope: &Scope,
                optb: <Option<T> as cubecl::prelude::CubeType>::ExpandType,
            ) -> <Option<T> as cubecl::prelude::CubeType>::ExpandType {
                {
                    match (self, optb) {
                        (a @ OptionExpand::Some(_), OptionExpand::None) => a,
                        (OptionExpand::None, b @ OptionExpand::Some(_)) => b,
                        _ => Option::__expand_new_None(scope),
                    }
                }
            }
            pub fn __expand_zip_method<U>(
                self,
                scope: &Scope,
                other: <Option<U> as cubecl::prelude::CubeType>::ExpandType,
            ) -> <Option<(T, U)> as cubecl::prelude::CubeType>::ExpandType
            where
                U: CubeType,
            {
                {
                    match (self, other) {
                        (OptionExpand::Some(a), OptionExpand::Some(b)) => {
                            let _arg_0 = (a, b);
                            Option::__expand_Some(scope, _arg_0)
                        }
                        _ => Option::__expand_new_None(scope),
                    }
                }
            }
        }
    }

    impl<T, U> ComptimeOption<(T, U)> {
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
        pub fn unzip(self) -> (Option<T>, Option<U>) {
            match self {
                Some((a, b)) => (Option::Some(a), Option::Some(b)),
                Option::None => (Option::None, Option::None),
            }
        }
    }

    impl<T: CubeType, U: CubeType> ComptimeOptionExpand<(T, U)> {
        pub fn __expand_unzip_method(
            self,
            scope: &Scope,
        ) -> <(Option<T>, Option<U>) as cubecl::prelude::CubeType>::ExpandType {
            {
                match self {
                    OptionExpand::Some((a, b)) => (
                        {
                            let _arg_0 = a;
                            Option::__expand_Some(scope, _arg_0)
                        },
                        {
                            let _arg_0 = b;
                            Option::__expand_Some(scope, _arg_0)
                        },
                    ),
                    OptionExpand::None => ({ Option::__expand_new_None(scope) }, {
                        Option::__expand_new_None(scope)
                    }),
                }
            }
        }
    }
}
