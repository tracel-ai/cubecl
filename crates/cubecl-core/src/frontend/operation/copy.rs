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
/// copy_bulk(input.as_slice(), shared, 16);
/// ```
pub fn copy_bulk<C: CubePrimitive>(_from: &Slice<C>, _to: &mut SliceMut<C>, _length: usize) {}

pub mod copy_bulk {
    use cubecl_ir::{CopyMemoryOperator, Memory};

    use crate::ir::{Instruction, Scope};

    use super::*;

    /// The expand function for [`copy_bulk()`]
    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        from: &SliceExpand<C, ReadOnly>,
        to: &mut SliceExpand<C, ReadWrite>,
        length: usize,
    ) {
        let source = from.__expand_as_ptr_method(scope).expand;
        let target = to.__expand_as_ptr_mut_method(scope).expand;

        scope.register(Instruction::no_out(Memory::CopyMemory(
            CopyMemoryOperator {
                source,
                target,
                len: length,
            },
        )));
    }
}
