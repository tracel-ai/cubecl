use cubecl_macros_internal::cube_op;
use pliron::derive::op_interface_impl;

use crate::{
    attributes::IndexAttr,
    interfaces::{ReadsMemory, WritesMemory},
    prelude::*,
};

#[pliron_op(name = "tma.load", format, attributes = (rank: IndexAttr), verifier = "succ")]
#[op_interfaces(AtLeastNOpdsInterface<4>)]
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
        op.set_attr_rank(ctx, rank.into());
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
        self.get_attr_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl WritesMemory for TmaLoadOp {
    fn writes_through_values(&self, ctx: &Context) -> Vec<Value> {
        vec![self.destination(ctx)]
    }
}

#[pliron_op(name = "tma.load_im2col", format, attributes = (rank: IndexAttr), verifier = "succ")]
#[op_interfaces(AtLeastNOpdsInterface<5>)]
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
        op.set_attr_rank(ctx, rank.into());
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
        let rank = self.get_attr_rank(ctx).unwrap().0;
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(3)
            .take(rank)
            .collect()
    }

    pub fn offsets(&self, ctx: &Context) -> Vec<Value> {
        let rank = self.get_attr_rank(ctx).unwrap().0;
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(3 + rank)
            .collect()
    }

    pub fn rank(&self, ctx: &Context) -> usize {
        self.get_attr_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl WritesMemory for TmaLoadIm2colOp {
    fn writes_through_values(&self, ctx: &Context) -> Vec<Value> {
        vec![self.destination(ctx)]
    }
}

#[pliron_op(name = "tma.store", format, attributes = (rank: IndexAttr), verifier = "succ")]
#[op_interfaces(AtLeastNOpdsInterface<4>)]
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
        op.set_attr_rank(ctx, rank.into());
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
        self.get_attr_rank(ctx).unwrap().0
    }
}

#[op_interface_impl]
impl ReadsMemory for TmaStoreOp {
    fn reads_through_values(&self, ctx: &Context) -> Vec<Value> {
        vec![self.source(ctx)]
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
