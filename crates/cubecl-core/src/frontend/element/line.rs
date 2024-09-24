use std::marker::PhantomData;

use super::CubePrimitive;

/// A contiguous list of elements.
pub struct Line<P: CubePrimitive> {
    _p: PhantomData<P>,
}
