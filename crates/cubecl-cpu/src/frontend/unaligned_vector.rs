use cubecl_core::ir::{IndexOperands, Instruction, Memory};
use cubecl_core::{self as cubecl, prelude::*};
use cubecl_core::{intrinsic, ir::Variable};

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
pub trait UnalignedVector<E: Scalar, N: Size>: CubeType {
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

type ArrayExpand<T> = NativeExpand<Array<T>>;
type SharedMemoryExpand<T> = NativeExpand<SharedMemory<T>>;

macro_rules! impl_unaligned_vector {
    ($type:ident) => {
        #[cube]
        impl<E: Scalar, N: Size> UnalignedVector<E, N> for $type<E> {
            fn unaligned_vector_read(&self, index: usize) -> Vector<E, N> {
                unaligned_vector_read::<E, N>(self.as_slice(), index)
            }

            fn unaligned_vector_write(&mut self, index: usize, value: Vector<E, N>) {
                unaligned_vector_write::<E, N>(self.as_mut_slice(), index, value)
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
fn unaligned_vector_read<E: Scalar, N: Size>(this: &[E], index: usize) -> Vector<E, N> {
    intrinsic!(|scope| {
        let list: Variable = this.__extract_list(scope);
        if !matches!(list.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        let vector_size = N::__expand_value(scope);
        let out = scope.create_local(Type::pointer(list.ty, list.address_space()));
        scope.register(Instruction::new(
            Memory::Index(IndexOperands {
                list: list,
                index: index.expand,
                vector_size: 0,
                unroll_factor: 1,
                checked: false,
            }),
            out,
        ));
        let mut out: NativeExpand<Vector<E, N>> = out.into();
        out.__expand_deref_method(scope)
    })
}

#[cube]
#[allow(unused_variables)]
fn unaligned_vector_write<E: Scalar, N: Size>(this: &mut [E], index: usize, value: Vector<E, N>) {
    intrinsic!(|scope| {
        let list: Variable = this.__extract_list(scope);
        if !matches!(list.ty, cubecl::ir::Type::Scalar(_)) {
            todo!("Unaligned reads are only allowed on scalar arrays for now");
        }
        let vector_size = N::__expand_value(scope);
        let out = scope.create_local(Type::pointer(list.ty, list.address_space()));
        scope.register(Instruction::new(
            Memory::Index(IndexOperands {
                list,
                index: index.expand,
                vector_size: 0,
                unroll_factor: 1,
                checked: false,
            }),
            out,
        ));
        let mut out: NativeExpand<Vector<E, N>> = out.into();
        out.__expand_assign_method(scope, value);
    })
}
