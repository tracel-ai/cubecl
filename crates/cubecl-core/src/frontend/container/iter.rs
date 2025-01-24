use cubecl_ir::ExpandElement;

use crate::{
    ir::{Branch, Item, RangeLoop, Scope},
    prelude::{index, CubeIndex, CubePrimitive, CubeType, ExpandElementTyped, Iterable},
};

use super::Array;

pub trait SizedContainer:
    CubeIndex<ExpandElementTyped<u32>, Output = Self::Item> + CubeType
{
    type Item: CubeType<ExpandType = ExpandElementTyped<Self::Item>>;

    /// Return the length of the container.
    fn len(val: &ExpandElement, context: &mut Scope) -> ExpandElement {
        // By default we use the expand len method of the Array type.
        let val: ExpandElementTyped<Array<Self::Item>> = val.clone().into();
        val.__expand_len_method(context).expand
    }
}

impl<T: SizedContainer> Iterable<T::Item> for ExpandElementTyped<T> {
    fn expand(
        self,
        context: &mut Scope,
        mut body: impl FnMut(&mut Scope, <T::Item as CubeType>::ExpandType),
    ) {
        let index_ty = Item::new(u32::as_elem(context));
        let len: ExpandElement = T::len(&self.expand, context);

        let mut child = context.child();
        let i = child.create_local_restricted(index_ty);

        let item = index::expand(&mut child, self, i.clone().into());
        body(&mut child, item);

        context.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start: 0u32.into(),
            end: *len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(
        self,
        _context: &mut Scope,
        _body: impl FnMut(&mut Scope, <T::Item as CubeType>::ExpandType),
    ) {
        unimplemented!("Can't unroll array iterator")
    }
}
