use super::CubeType;
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{Scope, VectorSize};

/// Type from which we can read/to which we can write values in cube functions.
#[allow(clippy::len_without_is_empty)]
#[cube(expand_base_traits = "SliceOperatorExpand<T>
    + IndexExpand<NativeExpand<usize>, Output = NativeExpand<T>>
    + IndexMutExpand<NativeExpand<usize>, Output = NativeExpand<T>>")]
pub trait List<T: CubePrimitive>:
    SliceOperator<T> + CubeIndex<usize, Output = T> + CubeIndexMut<usize, Output = T> + Vectorized
{
    fn len(&self) -> usize {
        unexpanded!();
    }
}

pub trait Vectorized: CubeType<ExpandType: VectorizedExpand> {
    fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
    fn __expand_vector_size(_scope: &Scope, this: &Self::ExpandType) -> VectorSize {
        this.vector_size()
    }
}

pub trait VectorizedExpand {
    fn vector_size(&self) -> VectorSize;
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.vector_size()
    }
}
