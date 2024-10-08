use crate::prelude::*;

/// Bulk copy `length` elements between two array-likes without intermediates.
///
/// # Arguments
///
/// `from` - The array/tensor/shared memory to copy from
/// `from_index` - The `from` index to start the copy from
/// `to` - The array/tensor/shared memory to copy to
/// `to_index` - The `to` index to copy the elements to
///
/// # Example
///
/// ```ignore
/// copy_bulk(input.as_slice(), idx, shared, shared_idx, 16);
/// ```
pub fn copy_bulk<C: CubePrimitive>(
    _from: &Slice<C>,
    _from_index: u32,
    _to: &mut SliceMut<C>,
    _to_index: u32,
    _length: u32,
) {
}

pub mod copy_bulk {
    use crate::ir::{CopyBulkOperator, Operator};

    use super::*;

    /// The expand function for [`copy_bulk`]
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
