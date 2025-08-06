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
    use crate::ir::{CopyMemoryBulkOperator, Instruction, Operator, Scope};

    use super::*;

    /// The expand function for [`copy_bulk()`]
    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        from: SliceExpand<C, ReadOnly>,
        from_index: ExpandElementTyped<u32>,
        to: SliceExpand<C, ReadWrite>,
        to_index: ExpandElementTyped<u32>,
        length: u32,
    ) {
        let (input, input_offset) = from.__to_raw_parts();
        let (to, to_offset) = to.__to_raw_parts();

        scope.register(Instruction::new(
            Operator::CopyMemoryBulk(CopyMemoryBulkOperator {
                out_index: to_index.expand.consume(),
                input,
                in_index: from_index.expand.consume(),
                len: length,
                offset_input: input_offset,
                offset_out: to_offset,
            }),
            to,
        ));
    }
}
