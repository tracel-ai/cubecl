use crate::{
    frontend::{CubeContext, CubeType},
    unexpanded,
};

use super::{CubePrimitive, ExpandElement, ExpandElementTyped, Init, UInt, Vectorized};

#[derive(Clone, Copy)]
/// Encapsulates a value to signify it must be used at compilation time rather than in the kernel
///
/// Use `Comptime<Option<T>>` to have an alternate runtime behaviour if the compilation time value is not present
pub struct Comptime<T> {
    pub(crate) inner: T,
}

/// Type that can be used within [Comptime].
pub trait ComptimeType: CubeType + Into<ExpandElement> {
    /// Create the expand type from the normal type.
    fn into_expand(self) -> Self::ExpandType;
}

impl ComptimeType for UInt {
    fn into_expand(self) -> Self::ExpandType {
        ExpandElementTyped::new(self.into())
    }
}

impl<T> Comptime<T> {
    /// Create a new Comptime. Useful when hardcoding values in
    /// Cube kernels. For instance:
    /// if Comptime::new(false) {...} never generates the inner code block
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Get the inner value of a Comptime. For instance:
    /// let c = Comptime::new(false);
    /// if Comptime::get(c) {...}
    pub fn get(_comptime: Self) -> T {
        unexpanded!()
    }

    /// Executes a closure on the comptime and returns a new comptime containing the value.
    pub fn map<R, F: Fn(T) -> R>(_comptime: Self, _closure: F) -> Comptime<R> {
        unexpanded!()
    }

    pub fn __expand_map<R, F: Fn(T) -> R>(inner: T, closure: F) -> R {
        closure(inner)
    }
}

impl<T: ComptimeType> Comptime<Option<T>> {
    /// Map a Comptime optional to a Comptime boolean that tell
    /// whether the optional contained a value
    pub fn is_some(comptime: Self) -> Comptime<bool> {
        Comptime::new(comptime.inner.is_some())
    }

    /// Return the inner value of the Comptime if it exists,
    /// otherwise tell how to compute it at runtime
    pub fn unwrap_or_else<F>(_comptime: Self, mut _alt: F) -> T
    where
        F: FnOnce() -> T,
    {
        unexpanded!()
    }

    /// Expanded version of unwrap_or_else
    pub fn __expand_unwrap_or_else<F>(
        context: &mut CubeContext,
        t: Option<T>,
        alt: F,
    ) -> <T as CubeType>::ExpandType
    where
        F: FnOnce(&mut CubeContext) -> T::ExpandType,
    {
        match t {
            Some(t) => t.into_expand(),
            None => alt(context),
        }
    }
}

impl<T: Clone + Init> CubeType for Comptime<T> {
    type ExpandType = T;
}

impl<T: Vectorized> Comptime<T> {
    pub fn vectorization(_state: &T) -> Comptime<UInt> {
        unexpanded!()
    }

    pub fn __expand_vectorization(_context: &mut CubeContext, state: T) -> UInt {
        state.vectorization_factor()
    }
}

impl<T: CubePrimitive + Into<ExpandElement>> Comptime<T> {
    pub fn runtime(_comptime: Self) -> T {
        unexpanded!()
    }

    pub fn __expand_runtime(_context: &mut CubeContext, inner: T) -> ExpandElementTyped<T> {
        let elem: ExpandElement = inner.into();
        elem.into()
    }
}

impl<T: core::ops::Add<T, Output = T>> core::ops::Add for Comptime<T> {
    type Output = Comptime<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Comptime::new(self.inner.add(rhs.inner))
    }
}

impl<T: core::ops::Sub<T, Output = T>> core::ops::Sub for Comptime<T> {
    type Output = Comptime<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Comptime::new(self.inner.sub(rhs.inner))
    }
}

impl<T: core::ops::Div<T, Output = T>> core::ops::Div for Comptime<T> {
    type Output = Comptime<T>;

    fn div(self, rhs: Self) -> Self::Output {
        Comptime::new(self.inner.div(rhs.inner))
    }
}

impl<T: core::ops::Mul<T, Output = T>> core::ops::Mul for Comptime<T> {
    type Output = Comptime<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Comptime::new(self.inner.mul(rhs.inner))
    }
}

impl<T: core::ops::Rem<T, Output = T>> core::ops::Rem for Comptime<T> {
    type Output = Comptime<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        Comptime::new(self.inner.rem(rhs.inner))
    }
}

impl<T: core::cmp::PartialOrd + core::cmp::PartialEq> core::cmp::PartialEq for Comptime<T> {
    fn eq(&self, other: &Self) -> bool {
        core::cmp::PartialEq::eq(&self.inner, &other.inner)
    }
}

impl<T: core::cmp::PartialOrd + core::cmp::PartialEq> core::cmp::PartialOrd for Comptime<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        core::cmp::PartialOrd::partial_cmp(&self.inner, &other.inner)
    }
}
