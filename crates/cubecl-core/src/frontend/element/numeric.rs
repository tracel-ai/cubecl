use std::num::NonZero;

use num_traits::NumCast;

use crate::compute::KernelLauncher;
use crate::ir::{Item, Variable};
use crate::prelude::Clamp;
use crate::Runtime;
use crate::{
    frontend::{index_assign, Abs, Max, Min, Remainder},
    unexpanded,
};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType},
    prelude::CubeIndexMut,
};

use super::{
    ArgSettings, ExpandElement, ExpandElementBaseInit, ExpandElementTyped, LaunchArg,
    LaunchArgExpand, Vectorized,
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
    + Vectorized
    + CubePrimitive
    + LaunchArgExpand<CompilationArg = ()>
    + ScalarArgSettings
    + ExpandElementBaseInit
    + Into<ExpandElementTyped<Self>>
    + CubeIndexMut<u32, Output = Self>
    + CubeIndexMut<ExpandElementTyped<u32>, Output = Self>
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
    const MAX: Self;
    const MIN: Self;

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
        _context: &mut CubeContext,
        val: ExpandElementTyped<i64>,
    ) -> <Self as CubeType>::ExpandType {
        let elem = Self::as_elem();
        let var: Variable = elem.constant_from_i64(val.constant().unwrap().as_i64());

        ExpandElement::Plain(var).into()
    }

    fn __expand_from_vec<const D: usize>(
        context: &mut CubeContext,
        vec: [u32; D],
    ) -> <Self as CubeType>::ExpandType {
        let new_var = context.create_local_binding(Item::vectorized(
            Self::as_elem(),
            NonZero::new(vec.len() as u8),
        ));
        let elem = Self::as_elem();

        for (i, element) in vec.iter().enumerate() {
            let var: Variable = elem.constant_from_i64(*element as i64);
            let expand = ExpandElement::Plain(var);

            index_assign::expand::<u32>(
                context,
                new_var.clone().into(),
                ExpandElementTyped::from_lit(i),
                expand.into(),
            );
        }

        new_var.into()
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
    elem: T,
}

impl<T: Numeric, R: Runtime> ArgSettings<R> for ScalarArg<T> {
    fn register(&self, launcher: &mut crate::compute::KernelLauncher<R>) {
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
