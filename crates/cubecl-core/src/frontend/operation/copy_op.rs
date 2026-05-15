use crate::prelude::*;

/// Bulk copy `length` elements between two array-likes without intermediates.
///
/// # Arguments
///
/// `from` - slice to copy from
/// `to` - slice to copy to
///
/// # Example
///
/// ```ignore
/// copy_bulk(input.as_slice(), shared, 16);
/// ```
pub fn copy_bulk<C: CubePrimitive>(_from: &[C], _to: &mut [C], _length: usize) {}

pub mod copy_bulk {
    use cubecl_ir::{CopyMemoryOperands, Memory};

    use crate::ir::{Instruction, Scope};

    use super::*;

    /// The expand function for [`copy_bulk()`]
    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        from: &SliceExpand<C>,
        to: &mut SliceExpand<C>,
        length: usize,
    ) {
        let source = unsafe { *from.__expand_as_ptr_method(scope) }.expand;
        let target = unsafe { *to.__expand_as_mut_ptr_method(scope) }.expand;

        scope.register(Instruction::no_out(Memory::CopyMemory(
            CopyMemoryOperands {
                source,
                target,
                len: length,
            },
        )));
    }
}

/// Copy one element between two array-likes without intermediates.
///
/// # Arguments
///
/// `from` - The reference to copy from
/// `to` - The reference to copy to
///
/// # Example
///
/// ```ignore
/// copy(input[0], shared[0]);
/// ```
pub fn copy<C: CubePrimitive>(_from: &C, _to: &mut C) {}

pub mod copy {
    use cubecl_ir::{CopyMemoryOperands, Memory};

    use crate::ir::{Instruction, Scope};

    use super::*;

    /// The expand function for [`copy_bulk()`]
    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        from: &NativeExpand<C>,
        to: &mut NativeExpand<C>,
    ) {
        let source = from.expand;
        let target = to.expand;

        scope.register(Instruction::no_out(Memory::CopyMemory(
            CopyMemoryOperands {
                source,
                target,
                len: 1,
            },
        )));
    }
}
