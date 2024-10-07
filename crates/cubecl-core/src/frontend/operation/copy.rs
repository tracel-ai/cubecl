use crate::prelude::{CubePrimitive, Slice, SliceMut};

pub fn copy_bulk<C: CubePrimitive>(
    _from: &Slice<C>,
    _from_index: u32,
    _to: &mut SliceMut<C>,
    _to_index: u32,
    _length: u32,
) {
}

pub mod copy_bulk {
    use crate::{
        ir::{CopyBulkOperator, Operator},
        prelude::{CubeContext, CubeType, ExpandElementTyped, Slice, SliceMut},
    };

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        from: ExpandElementTyped<Slice<C>>,
        from_index: ExpandElementTyped<u32>,
        to: ExpandElementTyped<SliceMut<C>>,
        to_index: ExpandElementTyped<u32>,
        length: u32,
    ) {
        context.register(Operator::CopyBulk(CopyBulkOperator {
            out: *to.expand,
            out_index: to_index.expand.consume(),
            input: from.expand.consume(),
            in_index: from_index.expand.consume(),
            len: length,
        }));
    }
}
