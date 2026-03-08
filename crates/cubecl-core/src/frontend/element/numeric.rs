use core::marker::PhantomData;

use cubecl_ir::{ConstantValue, ExpandElement};
use cubecl_runtime::runtime::Runtime;
use num_traits::NumCast;

use crate::ir::{Scope, Variable};
use crate::{CubeScalar, compute::KernelBuilder};
use crate::{compute::KernelLauncher, prelude::CompilationArg};
use crate::{
    frontend::{Abs, Remainder},
    unexpanded,
};
use crate::{
    frontend::{CubePrimitive, CubeType},
    prelude::InputScalar,
};

use super::{ExpandElementAssign, ExpandElementTyped, IntoRuntime, LaunchArg};

/// Type that encompasses both (unsigned or signed) integers and floats
/// Used in kernels that should work for both.
pub trait Numeric:
    Copy
    + Abs
    + Remainder
    + CubePrimitive
    + IntoRuntime
    + ExpandElementAssign
    + Into<ExpandElementTyped<Self>>
    + Into<ConstantValue>
    + num_traits::NumCast
    + num_traits::NumAssign
    + core::cmp::PartialOrd
    + core::cmp::PartialEq
    + core::fmt::Debug
    + Default
    + ScalarArgSettings
{
    fn min_value() -> Self;
    fn max_value() -> Self;

    fn __expand_min_value(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_type(scope).elem_type();
        let var = elem.min_variable();
        let expand = ExpandElement::Plain(var);
        expand.into()
    }

    fn __expand_max_value(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_type(scope).elem_type();
        let var = elem.max_variable();
        let expand = ExpandElement::Plain(var);
        expand.into()
    }

    /// Create a new constant numeric.
    ///
    /// Note: since this must work for both integer and float
    /// only the less expressive of both can be created (int)
    /// If a number with decimals is needed, use `Float::new`.
    ///
    /// This method panics when unexpanded. For creating an element
    /// with a val, use the new method of the sub type.
    fn from_int(val: i64) -> Self {
        <Self as NumCast>::from(val).unwrap()
    }

    /// Create a new constant numeric. Uses `i128` to be able to represent both signed integers, and
    /// `u64::MAX`.
    ///
    /// Note: since this must work for both integer and float
    /// only the less expressive of both can be created (int)
    /// If a number with decimals is needed, use `Float::new`.
    ///
    /// This method panics when unexpanded. For creating an element
    /// with a val, use the new method of the sub type.
    fn from_int_128(val: i128) -> Self {
        <Self as NumCast>::from(val).unwrap()
    }

    fn from_vec<const D: usize>(_vec: [u32; D]) -> Self {
        unexpanded!()
    }

    fn __expand_from_int(
        scope: &mut Scope,
        val: ExpandElementTyped<i64>,
    ) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_type(scope).elem_type();
        let var: Variable = elem.constant(val.constant().unwrap());

        ExpandElement::Plain(var).into()
    }
}

/// Similar to [`ArgSettings`], however only for scalar types that don't depend on the [Runtime]
/// trait.
pub trait ScalarArgSettings: Send + Sync + CubePrimitive {
    /// Register the information to the [`KernelLauncher`].
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>);
    fn expand_scalar(
        _: &ScalarCompilationArg<Self>,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Self> {
        builder
            .scalar(Self::as_type(&builder.scope).storage_type())
            .into()
    }
}

impl<E: CubeScalar> ScalarArgSettings for E {
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_scalar(*self);
    }
}

impl ScalarArgSettings for usize {
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>) {
        let value = InputScalar::new(*self, launcher.settings.address_type.unsigned_type());
        InputScalar::register(value, launcher);
    }
}

impl ScalarArgSettings for isize {
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>) {
        let value = InputScalar::new(*self, launcher.settings.address_type.signed_type());
        InputScalar::register(value, launcher);
    }
}

#[derive(new, Clone, Copy)]
pub struct ScalarArg<T: ScalarArgSettings> {
    pub elem: T,
}

#[derive(new, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScalarCompilationArg<T: ScalarArgSettings> {
    _ty: PhantomData<T>,
}

impl<T: ScalarArgSettings> Eq for ScalarCompilationArg<T> {}
impl<T: ScalarArgSettings> core::hash::Hash for ScalarCompilationArg<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self._ty.hash(state);
    }
}
impl<T: ScalarArgSettings> core::fmt::Debug for ScalarCompilationArg<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Scalar")
    }
}

impl<T: ScalarArgSettings> CompilationArg for ScalarCompilationArg<T> {}

impl<T: ScalarArgSettings> LaunchArg for T {
    type RuntimeArg<R: Runtime> = ScalarArg<T>;
    type CompilationArg = ScalarCompilationArg<T>;

    fn compilation_arg<R: Runtime>(_runtime_arg: &Self::RuntimeArg<R>) -> Self::CompilationArg {
        ScalarCompilationArg::new()
    }

    fn register<R: Runtime>(arg: Self::RuntimeArg<R>, launcher: &mut KernelLauncher<R>) {
        arg.elem.register(launcher);
    }

    fn expand(
        arg: &ScalarCompilationArg<T>,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Self> {
        T::expand_scalar(arg, builder)
    }
}
