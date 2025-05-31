use cubecl_ir::ExpandElement;
use num_traits::NumCast;

use crate::Runtime;
use crate::compute::KernelLauncher;
use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::{Scope, Variable};
use crate::prelude::Clamp;
use crate::{
    frontend::{Abs, Max, Min, Remainder},
    unexpanded,
};

use super::{
    ArgSettings, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime, LaunchArg, LaunchArgExpand,
};

/// Type that encompasses both (unsigned or signed) integers and floats
/// Used in kernels that should work for both.
pub trait Numeric:
    Copy
    + Abs
    + Max
    + Min
    + Clamp
    + Remainder
    + CubePrimitive
    + IntoRuntime
    + LaunchArgExpand<CompilationArg = ()>
    + ScalarArgSettings
    + ExpandElementIntoMut
    + Into<ExpandElementTyped<Self>>
    + num_traits::NumCast
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::cmp::PartialOrd
    + std::cmp::PartialEq
{
    fn min_value() -> Self;
    fn max_value() -> Self;

    fn __expand_min_value(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_elem(scope);
        let var = elem.min_variable();
        let expand = ExpandElement::Plain(var);
        expand.into()
    }

    fn __expand_max_value(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_elem(scope);
        let var = elem.max_variable();
        let expand = ExpandElement::Plain(var);
        expand.into()
    }

    /// Create a new constant numeric.
    ///
    /// Note: since this must work for both integer and float
    /// only the less expressive of both can be created (int)
    /// If a number with decimals is needed, use Float::new.
    ///
    /// This method panics when unexpanded. For creating an element
    /// with a val, use the new method of the sub type.
    fn from_int(val: i64) -> Self {
        <Self as NumCast>::from(val).unwrap()
    }

    fn from_vec<const D: usize>(_vec: [u32; D]) -> Self {
        unexpanded!()
    }

    fn __expand_from_int(
        scope: &mut Scope,
        val: ExpandElementTyped<i64>,
    ) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_elem(scope);
        let var: Variable = elem.constant_from_i64(val.constant().unwrap().as_i64());

        ExpandElement::Plain(var).into()
    }
}

/// Similar to [ArgSettings], however only for scalar types that don't depend on the [Runtime]
/// trait.
pub trait ScalarArgSettings: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>);
}

#[derive(new)]
pub struct ScalarArg<T: Numeric> {
    pub elem: T,
}

impl<T: Numeric, R: Runtime> ArgSettings<R> for ScalarArg<T> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        self.elem.register(launcher);
    }
}

impl<T: Numeric> LaunchArg for T {
    type RuntimeArg<'a, R: Runtime> = ScalarArg<T>;

    fn compilation_arg<'a, R: Runtime>(
        _runtime_arg: &'a Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
    }
}
