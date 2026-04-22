use core::ops::{Deref, DerefMut};

use super::CubeType;
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{Scope, VectorSize};

/// Type from which we can read values in cube functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[cube(expand_base_traits = "SliceOperatorExpand<T>")]
pub trait List<T: CubePrimitive>: SliceOperator<T> + Vectorized + Deref<Target = [T]> {
    #[allow(unused)]
    fn read(&self, index: usize) -> &T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, index: usize) -> &T {
        unexpanded!()
    }

    #[allow(unused)]
    fn len(&self) -> usize {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[allow(clippy::mut_from_ref)]
#[cube(expand_base_traits = "SliceMutOperatorExpand<T>")]
pub trait ListMut<T: CubePrimitive>:
    List<T> + SliceMutOperator<T> + DerefMut<Target = [T]>
{
    #[allow(unused)]
    fn write(&self, index: usize) -> &mut T {
        unexpanded!()
    }
}

pub trait Vectorized: CubeType<ExpandType: VectorizedExpand> {
    fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
    fn __expand_vector_size(_scope: &Scope, this: Self::ExpandType) -> VectorSize {
        this.vector_size()
    }
}

pub trait VectorizedExpand {
    fn vector_size(&self) -> VectorSize;
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.vector_size()
    }
}
