use cubecl_core::intrinsic;
use cubecl_core::ir::{IndexAssignOperator, IndexOperator, Instruction, Operator};
use cubecl_core::{self as cubecl, prelude::*};

/// An extension trait for expanding the cubecl frontend with the ability to
/// request unaligned vector reads and writes
///
/// Typically in cubecl, a buffer is declared as having a certain vector size
/// at kernel compilation time. The buffer can then be indexed to produce
/// vectors that are aligned to the `vector_size`.
///
/// This trait allows the user to request a `vector_read` from a buffer where the
/// start of the read is not aligned to the `vector_read` requested.
///
/// As an example, imagine a buffer of scalar length 4. With `vector_size` = 1,
/// this could be illustrated like so
/// [1, 2, 3, 4]
///
/// Imagine the same buffer, now with `vector_size` = 2.
/// [[1, 2], [3, 4]]
///
/// Vectors can now be accessed from this buffer, but only those that that are aligned
/// with the `vector_size`. I.e. we can get the vectors [1, 2] or [3, 4], but not [2, 3]
///
/// This trait allows you to treat the buffer as having no `vector_size` = 1, yet asking
/// for a vector of some kernel-compile-time known length at some offset in the buffer.
/// I.e. if for the buffer `buf = [1, 2, 3, 4]`, `buf.unaligned_vector_read(1, 2)`
/// will produce the vector `[2, 3]`.
#[cube]
pub trait UnalignedVector<E: Scalar, N: Size>: CubeType + Sized {
    /// Perform an unchecked read of a vector of the given length at the given index
    ///
    /// # Safety
    /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure `index..index+vector_size` is
    /// always in bounds
    fn unaligned_vector_read(&self, index: usize) -> Vector<E, N>;

    /// Perform an unchecked write of a vector of the given length at the given index
    ///
    /// # Safety
    /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure `index..index+vector_size` is
    /// always in bounds
    fn unaligned_vector_write(&mut self, index: usize, value: Vector<E, N>);
}

macro_rules! impl_unaligned_vector {
    ($type:ident) => {
        paste::paste! {
            type [<$type Expand>]<E> = NativeExpand<$type<E>>;
        }
        #[cube]
        impl<E: Scalar, N: Size> UnalignedVector<E, N> for $type<E> {
            fn unaligned_vector_read(&self, index: usize) -> Vector<E, N> {
                unaligned_vector_read::<$type<E>, E, N>(self, index)
            }

            fn unaligned_vector_write(&mut self, index: usize, value: Vector<E, N>) {
                unaligned_vector_write::<$type<E>, E, N>(self, index, value)
            }
        }
    };
}

impl_unaligned_vector!(Array);
impl_unaligned_vector!(Tensor);
impl_unaligned_vector!(SharedMemory);

// TODO: Maybe impl unaligned IO on slices?
// The last dimension will have to be contiguous for this to make sense,
// as the unaligned IO isn't gather / scatter from arbitrary memory locations
// and still needs the loaded elements to be contiguous

#[cube]
#[allow(unused_variables)]
fn unaligned_vector_read<T: CubeType<ExpandType = NativeExpand<T>>, E: Scalar, N: Size>(
    this: &T,
    index: usize,
) -> Vector<E, N> {
    intrinsic!(|scope| {
        if !matches!(this.expand.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        let vector_size = N::__expand_value(scope);
        let out = scope.create_local(this.expand.ty.with_vector_size(vector_size));
        scope.register(Instruction::new(
            Operator::UncheckedIndex(IndexOperator {
                list: *this.expand,
                index: index.expand.consume(),
                vector_size: 0,
                unroll_factor: 1,
            }),
            *out,
        ));
        out.into()
    })
}

#[cube]
#[allow(unused_variables)]
fn unaligned_vector_write<T: CubeType<ExpandType = NativeExpand<T>>, E: Scalar, N: Size>(
    this: &mut T,
    index: usize,
    value: Vector<E, N>,
) {
    intrinsic!(|scope| {
        if !matches!(this.expand.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        let vector_size = N::__expand_value(scope);
        scope.register(Instruction::new(
            Operator::UncheckedIndexAssign(IndexAssignOperator {
                index: index.expand.consume(),
                value: value.expand.consume(),
                vector_size: 0,
                unroll_factor: 1,
            }),
            *this.expand,
        ));
    })
}
