use cubecl_ir::{ConstantValue, Variable};
use cubecl_runtime::runtime::Runtime;
use num_traits::{NumCast, One, Zero};

use crate::compute::KernelLauncher;
use crate::{IntoRuntime, ScalarArgType, compute::KernelBuilder};
use crate::{
    frontend::{Abs, ModFloor, VectorSum},
    unexpanded,
};
use crate::{
    frontend::{CubePrimitive, CubeType},
    prelude::InputScalar,
};
use crate::{ir::Scope, prelude::Scalar};

use super::{LaunchArg, NativeAssign, NativeExpand};

/// Type that encompasses both (unsigned or signed) integers and floats
/// Used in kernels that should work for both.
pub trait Numeric:
    Copy
    + Abs
    + VectorSum
    + ModFloor
    + Scalar
    + NativeAssign
    + Into<NativeExpand<Self>>
    + Into<ConstantValue>
    + num_traits::NumCast
    + num_traits::NumAssign
    + core::cmp::PartialOrd
    + core::cmp::PartialEq
    + core::fmt::Debug
    + bytemuck::Zeroable
{
    fn min_value() -> Self;
    fn max_value() -> Self;

    fn __expand_min_value(scope: &Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::__expand_as_type(scope).elem_type();
        let var = elem.min_variable();
        var.into()
    }

    fn __expand_max_value(scope: &Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::__expand_as_type(scope).elem_type();
        let var = elem.max_variable();
        var.into()
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

    fn __expand_from_int(scope: &Scope, val: NativeExpand<i64>) -> <Self as CubeType>::ExpandType {
        let elem = Self::__expand_as_type(scope).elem_type();
        let var: Variable = elem.constant(val.constant().unwrap());

        var.into()
    }
}

/// Similar to [`ArgSettings`], however only for scalar types that don't depend on the [Runtime]
/// trait.
pub trait ScalarArgSettings: Send + Sync + CubePrimitive {
    /// Register the information to the [`KernelLauncher`].
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>);
    fn expand_scalar(builder: &mut KernelBuilder) -> NativeExpand<Self> {
        builder
            .scalar(Self::__expand_as_type(&builder.scope).storage_type())
            .into()
    }
}

impl<E: ScalarArgType> ScalarArgSettings for E {
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

macro_rules! impl_scalar_launch {
    ($ty: ty) => {
        impl LaunchArg for $ty {
            type RuntimeArg<R: Runtime> = $ty;
            type CompilationArg = ();

            fn register<R: Runtime>(arg: Self::RuntimeArg<R>, launcher: &mut KernelLauncher<R>) {
                arg.register(launcher);
            }

            fn expand(_: &(), builder: &mut KernelBuilder) -> NativeExpand<Self> {
                <$ty>::expand_scalar(builder)
            }
        }
    };
}
pub(crate) use impl_scalar_launch;

pub trait ZeroExpand: CubeType + Zero {
    fn __expand_zero(scope: &Scope) -> Self::ExpandType;
}

pub trait OneExpand: CubeType + One {
    fn __expand_one(scope: &Scope) -> Self::ExpandType;
}

impl<T: CubeType + Zero + IntoRuntime> ZeroExpand for T {
    fn __expand_zero(scope: &Scope) -> Self::ExpandType {
        T::zero().__expand_runtime_method(scope)
    }
}

impl<T: CubeType + One + IntoRuntime> OneExpand for T {
    fn __expand_one(scope: &Scope) -> Self::ExpandType {
        T::one().__expand_runtime_method(scope)
    }
}
