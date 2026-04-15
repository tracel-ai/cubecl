use core::marker::PhantomData;

use crate::prelude::*;

#[derive(PartialEq, Eq)]
pub struct Ref<T: ?Sized> {
    _ty: PhantomData<*const T>,
}

impl<T: ?Sized> Copy for Ref<T> {}
impl<T: ?Sized> Clone for Ref<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> CubeType for Ref<T> {
    type ExpandType = NativeExpand<Ref<T>>;
}

impl<T> NativeAssign for Ref<T> {}

impl<T> NativeExpand<Ref<T>> {
    pub fn __expand_deref_method(self, _: &mut Scope) -> NativeExpand<T> {
        NativeExpand::<T> {
            expand: self.expand,
            _type: PhantomData,
        }
    }
}
