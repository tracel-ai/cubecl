use core::ops::RangeInclusive;

use alloc::format;
use cubecl_macros_internal::cube_op;
use pliron::{derive::op_interface_impl, printable::Printable, verify_err};
use thiserror::Error;

use crate::{
    AddressSpace,
    attributes::IndexAttr,
    interfaces::{MemoryEffect, MemoryEffects, TypeExt},
    prelude::*,
    types::{
        PointerType,
        barrier::{BarrierLevel, BarrierType},
        cuda::TensorMapType,
    },
};

#[derive(Error, Debug)]
pub enum TmaOpError {
    #[error("[TmaOp]: Invalid address space for {_0}. Expected {_1}, got {_2}.")]
    InvalidAddressSpace(&'static str, &'static str, AddressSpace),
    #[error("[TmaOp]: Unsupported rank {_0}, expected rank to be in range {_1:?}.")]
    UnsupportedRank(usize, RangeInclusive<usize>),
}

fn expected_barrier_ty(ctx: &Context) -> TypeHandle {
    PointerType::get(
        ctx,
        BarrierType::get(ctx, BarrierLevel::Cube).into(),
        AddressSpace::Shared,
    )
    .to_handle()
}

#[pliron_op(name = "tma.load", format, attributes = (tma_load_rank: IndexAttr))]
#[op_interfaces(AtLeastNOpdsInterface<4>, OperandNOfType<0, PointerType>, OperandNOfType<1, TensorMapType>, OperandNOfType<2, PointerType>)]
pub struct TmaLoadOp;

impl TmaLoadOp {
    pub fn new(
        ctx: &mut Context,
        barrier: Value,
        tensor_map: Value,
        destination: Value,
        indices: Vec<Value>,
    ) -> Self {
        let rank = indices.len();
        let mut operands = vec![barrier, tensor_map, destination];
        operands.extend(indices);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_tma_load_rank(ctx, rank.into());
        op
    }

    pub fn barrier(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn tensor_map(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(3).collect()
    }

    pub fn rank(&self, ctx: &Context) -> usize {
        self.get_attr_tma_load_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl MemoryEffects for TmaLoadOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::Write(self.destination(ctx))]
    }
}

impl Verify for TmaLoadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let barrier_ty = self.barrier(ctx).get_type(ctx).as_ptr(ctx);
        let dest_ty = self.barrier(ctx).get_type(ctx).as_ptr(ctx);

        if !barrier_ty.inner.deref(ctx).is::<BarrierType>() {
            let expected = expected_barrier_ty(ctx).deref(ctx);
            return verify_err!(
                loc,
                OperandNOfTypeError::AllOperandsOfTypeVerifyErr(
                    format!("{} {}", expected.get_type_id(), expected.disp(ctx)),
                    format!("{} {}", barrier_ty.get_type_id(), barrier_ty.disp(ctx))
                )
            )?;
        }

        if dest_ty.address_space != AddressSpace::Shared {
            return verify_err!(
                loc,
                TmaOpError::InvalidAddressSpace("destination", "Shared", dest_ty.address_space)
            )?;
        }

        if !(1..=5).contains(&self.rank(ctx)) {
            return verify_err!(loc, TmaOpError::UnsupportedRank(self.rank(ctx), 1..=5))?;
        }

        Ok(())
    }
}

#[pliron_op(name = "tma.load_im2col", format, attributes = (tma_load_im2col_rank: IndexAttr))]
#[op_interfaces(AtLeastNOpdsInterface<5>, OperandNOfType<0, PointerType>, OperandNOfType<1, TensorMapType>, OperandNOfType<2, PointerType>)]
pub struct TmaLoadIm2colOp;

impl TmaLoadIm2colOp {
    pub fn new(
        ctx: &mut Context,
        barrier: Value,
        tensor_map: Value,
        destination: Value,
        indices: Vec<Value>,
        offsets: Vec<Value>,
    ) -> Self {
        let rank = indices.len();
        let mut operands = vec![barrier, tensor_map, destination];
        operands.extend(indices);
        operands.extend(offsets);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_tma_load_im2col_rank(ctx, rank.into());
        op
    }

    pub fn barrier(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn tensor_map(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn indices(&self, ctx: &Context) -> Vec<Value> {
        let rank = self.get_attr_tma_load_im2col_rank(ctx).unwrap().0;
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(3)
            .take(rank)
            .collect()
    }

    pub fn offsets(&self, ctx: &Context) -> Vec<Value> {
        let rank = self.get_attr_tma_load_im2col_rank(ctx).unwrap().0;
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(3 + rank)
            .collect()
    }

    pub fn rank(&self, ctx: &Context) -> usize {
        self.get_attr_tma_load_im2col_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl MemoryEffects for TmaLoadIm2colOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::Write(self.destination(ctx))]
    }
}

impl Verify for TmaLoadIm2colOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let barrier_ty = self.barrier(ctx).get_type(ctx).as_ptr(ctx);
        let dest_ty = self.barrier(ctx).get_type(ctx).as_ptr(ctx);

        if !barrier_ty.inner.deref(ctx).is::<BarrierType>() {
            let expected = expected_barrier_ty(ctx).deref(ctx);
            return verify_err!(
                loc,
                OperandNOfTypeError::AllOperandsOfTypeVerifyErr(
                    format!("{} {}", expected.get_type_id(), expected.disp(ctx)),
                    format!("{} {}", barrier_ty.get_type_id(), barrier_ty.disp(ctx))
                )
            )?;
        }

        if dest_ty.address_space != AddressSpace::Shared {
            return verify_err!(
                loc,
                TmaOpError::InvalidAddressSpace("destination", "Shared", dest_ty.address_space)
            )?;
        }

        if !(3..=5).contains(&self.rank(ctx)) {
            return verify_err!(loc, TmaOpError::UnsupportedRank(self.rank(ctx), 3..=5))?;
        }

        Ok(())
    }
}

#[pliron_op(name = "tma.store", format, attributes = (tma_store_rank: IndexAttr))]
#[op_interfaces(AtLeastNOpdsInterface<4>, OperandNOfType<0, PointerType>, OperandNOfType<1, TensorMapType>)]
pub struct TmaStoreOp;

impl TmaStoreOp {
    pub fn new(ctx: &mut Context, source: Value, tensor_map: Value, indices: Vec<Value>) -> Self {
        let rank = indices.len();
        let mut operands = vec![source, tensor_map];
        operands.extend(indices);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_tma_store_rank(ctx, rank.into());
        op
    }

    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn tensor_map(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(2).collect()
    }

    pub fn rank(&self, ctx: &Context) -> usize {
        self.get_attr_tma_store_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl MemoryEffects for TmaStoreOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::Read(self.source(ctx))]
    }
}

impl Verify for TmaStoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let src_ty = self.source(ctx).get_type(ctx).as_ptr(ctx);

        if src_ty.address_space != AddressSpace::Shared {
            return verify_err!(
                loc,
                TmaOpError::InvalidAddressSpace("source", "Shared", src_ty.address_space)
            )?;
        }

        if !(1..=5).contains(&self.rank(ctx)) {
            return verify_err!(loc, TmaOpError::UnsupportedRank(self.rank(ctx), 1..=5))?;
        }

        Ok(())
    }
}

#[cube_op(name = "tma.commit_group")]
#[result_ty(none)]
pub struct CommitGroupOp {}

#[cube_op(name = "tma.wait_group")]
#[result_ty(none)]
pub struct WaitGroupOp {
    pub max_pending: IndexAttr,
}

#[cube_op(name = "tma.wait_group_read")]
#[result_ty(none)]
pub struct WaitGroupReadOp {
    pub max_pending: IndexAttr,
}
