use cubecl_macros_internal::cube_op;

use crate::{
    CanMaterialize, Pure,
    attributes::IndexAttr,
    prelude::*,
    types::{
        PointerType,
        spirv::{ClampMode, TensorLayoutType},
    },
};

#[pliron_op(
    name = "matrix.spirv.load_tensor",
    operands = (out_mat: PointerType, buffer, layout: TensorLayoutType),
    format,
    verifier = "succ"
)]
#[op_traits(CanMaterialize)]
pub struct LoadTensorOp;

impl LoadTensorOp {
    pub fn new(
        ctx: &mut Context,
        out_mat: Value,
        buffer: Value,
        layout: Value,
        view: Option<Value>,
    ) -> Self {
        let mut operands = vec![out_mat, buffer, layout];
        operands.extend(view);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        Self { op }
    }

    pub fn out_mat(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn buffer(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn layout(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn view(&self, ctx: &Context) -> Option<Value> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() > 3 {
            Some(op.get_operand(3))
        } else {
            None
        }
    }
}

#[pliron_op(name = "matrix.spirv.store_tensor", format, verifier = "succ")]
#[op_traits(CanMaterialize)]
pub struct StoreTensorOp;

impl StoreTensorOp {
    pub fn new(
        ctx: &mut Context,
        buffer: Value,
        matrix: Value,
        layout: Value,
        view: Option<Value>,
    ) -> Self {
        let mut operands = vec![buffer, matrix, layout];
        operands.extend(view);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        Self { op }
    }

    pub fn buffer(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn matrix(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn layout(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn view(&self, ctx: &Context) -> Option<Value> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() > 3 {
            Some(op.get_operand(3))
        } else {
            None
        }
    }
}

#[pliron_op(name = "spirv.create_layout", format, attributes = (spirv_create_layout_rank: IndexAttr), verifier = "succ")]
#[op_interfaces(NResultsInterface<1>, OneResultInterface)]
#[op_traits(CanMaterialize, Pure)]
pub struct CreateLayoutOp;

impl CreateLayoutOp {
    pub fn new(
        ctx: &mut Context,
        shape: Vec<Value>,
        strides: Option<Vec<Value>>,
        clamp_mode: ClampMode,
    ) -> Self {
        let rank = shape.len();
        let out_ty = TensorLayoutType::get(ctx, rank, clamp_mode);
        let mut operands = shape;
        operands.extend(strides.into_iter().flatten());
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![out_ty.into()],
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_spirv_create_layout_rank(ctx, rank.into());
        op
    }

    pub fn shape(&self, ctx: &Context) -> Vec<Value> {
        let rank = self.rank(ctx);
        let op = self.get_operation().deref(ctx);
        op.operands().take(rank).collect()
    }

    pub fn strides(&self, ctx: &Context) -> Option<Vec<Value>> {
        let rank = self.rank(ctx);
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() > rank {
            Some(op.operands().skip(rank).collect())
        } else {
            None
        }
    }

    pub fn rank(&self, ctx: &Context) -> usize {
        self.get_attr_spirv_create_layout_rank(ctx).unwrap().0
    }
}

#[cube_op(name = "cube.spirv.create_view")]
#[result_ty(argument)]
#[op_traits(CanMaterialize, Pure)]
pub struct CreateViewOp {}

#[pliron_op(
    name = "spirv.slice_layout",
    format,
    attributes = (spirv_slice_layout_rank: IndexAttr),
    verifier = "succ"
)]
#[op_interfaces(NResultsInterface<1>, OneResultInterface, OperandSegmentInterface)]
#[op_traits(CanMaterialize, Pure)]
pub struct SliceOp;

impl SliceOp {
    pub fn new(ctx: &mut Context, layout: Value, offsets: Vec<Value>, shape: Vec<Value>) -> Self {
        let (operands, segment_sizes) =
            Self::compute_segment_sizes(vec![vec![layout], offsets, shape]);
        let out_ty = layout.get_type(ctx);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![out_ty],
                operands,
                vec![],
                0,
            ),
        };
        op.set_operand_segment_sizes(ctx, segment_sizes);
        op
    }

    pub fn layout(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn offsets(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    pub fn shape(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }
}
