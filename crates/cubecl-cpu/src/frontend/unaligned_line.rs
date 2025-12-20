use cubecl_core::intrinsic;
use cubecl_core::ir::{IndexAssignOperator, IndexOperator, Instruction, Operator};
use cubecl_core::{self as cubecl, prelude::*};

/// An extension trait for expanding the cubecl frontend with the ability to
/// request unaligned line reads and writes
///
/// Typically in cubecl, a buffer is declared as having a certain line size
/// at kernel compilation time. The buffer can then be indexed to produce
/// lines that are aligned to the line_size.
///
/// This trait allows the user to request a line_read from a buffer where the
/// start of the read is not aligned to the line_read requested.
///
/// As an example, imagine a buffer of scalar length 4. With line_size = 1,
/// this could be illustrated like so
/// [1, 2, 3, 4]
///
/// Imagine the same buffer, now with line_size = 2.
/// [[1, 2], [3, 4]]
///
/// Lines can now be accessed from this buffer, but only those that that are aligned
/// with the line_size. I.e. we can get the lines [1, 2] or [3, 4], but not [2, 3]
///
/// This trait allows you to treat the buffer as having no line_size = 1, yet asking
/// for a line of some kernel-compile-time known length at some offset in the buffer.
/// I.e. if for the buffer `buf = [1, 2, 3, 4]`, `buf.unaligned_line_read(1, 2)`
/// will produce the line `[2, 3]`.
#[cube]
pub trait UnalignedLine<E: CubePrimitive>: CubeType + Sized {
    /// Perform an unchecked read of a line of the given length at the given index
    ///
    /// # Safety
    /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index..index+line_size is
    /// always in bounds
    fn unaligned_line_read(&self, index: usize, #[comptime] line_size: LineSize) -> Line<E>;

    /// Perform an unchecked write of a line of the given length at the given index
    ///
    /// # Safety
    /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index..index+line_size is
    /// always in bounds
    fn unaligned_line_write(&mut self, index: usize, value: Line<E>);
}

macro_rules! impl_unaligned_line {
    ($type:ident) => {
        paste::paste! {
            type [<$type Expand>]<E> = ExpandElementTyped<$type<E>>;
        }
        #[cube]
        impl<E: CubePrimitive> UnalignedLine<E> for $type<E> {
            fn unaligned_line_read(
                &self,
                index: usize,
                #[comptime] line_size: LineSize,
            ) -> Line<E> {
                unaligned_line_read::<$type<E>, E>(self, index, line_size)
            }

            fn unaligned_line_write(&mut self, index: usize, value: Line<E>) {
                unaligned_line_write::<$type<E>, E>(self, index, value)
            }
        }
    };
}

impl_unaligned_line!(Array);
impl_unaligned_line!(Tensor);
impl_unaligned_line!(SharedMemory);

// TODO: Maybe impl unaligned IO on slices?
// The last dimension will have to be contiguous for this to make sense,
// as the unaligned IO isn't gather / scatter from arbitrary memory locations
// and still needs the loaded elements to be contiguous

#[cube]
#[allow(unused_variables)]
fn unaligned_line_read<T: CubeType<ExpandType = ExpandElementTyped<T>>, E: CubePrimitive>(
    this: &T,
    index: usize,
    #[comptime] line_size: LineSize,
) -> Line<E> {
    intrinsic!(|scope| {
        if !matches!(this.expand.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        let out = scope.create_local(this.expand.ty.line(line_size));
        scope.register(Instruction::new(
            Operator::UncheckedIndex(IndexOperator {
                list: *this.expand,
                index: index.expand.consume(),
                line_size: 0,
                unroll_factor: 1,
            }),
            *out,
        ));
        out.into()
    })
}

#[cube]
#[allow(unused_variables)]
fn unaligned_line_write<T: CubeType<ExpandType = ExpandElementTyped<T>>, E: CubePrimitive>(
    this: &mut T,
    index: usize,
    value: Line<E>,
) {
    intrinsic!(|scope| {
        if !matches!(this.expand.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        scope.register(Instruction::new(
            Operator::UncheckedIndexAssign(IndexAssignOperator {
                index: index.expand.consume(),
                value: value.expand.consume(),
                line_size: 0,
                unroll_factor: 1,
            }),
            *this.expand,
        ));
    })
}
