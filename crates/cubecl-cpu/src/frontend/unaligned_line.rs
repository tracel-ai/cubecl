// use crate::frontend::unaligned_line::cubecl::intrinsic;
use cubecl_core::intrinsic;
use cubecl_core::ir::{IndexAssignOperator, IndexOperator, Instruction, Operator};
use cubecl_core::{self as cubecl, prelude::*};

#[cube]
pub trait UnalignedLine<E: CubePrimitive>:
    CubeType<ExpandType = ExpandElementTyped<Self>> + Sized
{
    fn unaligned_line_read(&self, index: u32, #[comptime] line_size: u32) -> Line<E>;

    fn unaligned_line_write(&mut self, index: u32, value: Line<E>);
}

macro_rules! impl_unaligned_line {
    ($type:ident) => {
        paste::paste! {
            type [<$type Expand>]<E> = ExpandElementTyped<$type<E>>;
        }
        #[cube]
        impl<E: CubePrimitive> UnalignedLine<E> for $type<E> {
            fn unaligned_line_read(&self, index: u32, #[comptime] line_size: u32) -> Line<E> {
                unaligned_line_read::<$type<E>, E>(self, index, line_size)
            }

            fn unaligned_line_write(&mut self, index: u32, value: Line<E>) {
                unaligned_line_write::<$type<E>, E>(self, index, value)
            }
        }
    };
}

impl_unaligned_line!(Array);
impl_unaligned_line!(Tensor);
impl_unaligned_line!(SharedMemory);

#[cube]
#[allow(unused_variables)]
fn unaligned_line_read<T: CubeType<ExpandType = ExpandElementTyped<T>>, E: CubePrimitive>(
    this: &T,
    index: u32,
    #[comptime] line_size: u32,
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
    index: u32,
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
