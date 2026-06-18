use core::cell::Ref;

use alloc::string::String;

use cubecl_macros_internal::{const_eval, cube_op};
use derive_more::From;
use derive_new::new;
use pliron::{
    builtin::attributes::{StringAttr, TypeAttr},
    derive::pliron_attr,
    r#type::{TypeHandle, type_cast},
};

use crate::{
    Builtin,
    attributes::{BoolAttr, IndexAttr},
    dialect::{pure_binop, pure_unop},
    interfaces::{AggregateType, Pure, erasable},
    prelude::*,
    types::scalar::IndexType,
};

#[cube_op(name = "cube.copy")]
#[result_ty(same_as = value)]
#[op_interfaces(Pure)]
pub struct CopyOp {
    pub value: Value,
}

#[pliron_op(name = "aggregate.construct", format, verifier = "succ")]
#[op_interfaces(NResultsInterface<1>, OneResultInterface)]
pub struct AggregateConstructOp;
erasable!(AggregateConstructOp);

impl AggregateConstructOp {
    pub fn new(ctx: &mut Context, ty: TypeHandle, values: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ty],
            values,
            vec![],
            0,
        );
        Self { op }
    }

    pub fn values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

#[cube_op(name = "aggregate.extract")]
#[result_ty(from_inputs = aggregate_extract_type)]
pub struct AggregateExtractOp {
    pub aggregate: Value,
    pub field: IndexAttr,
}
erasable!(AggregateExtractOp);

fn aggregate_extract_type(ctx: &Context, aggregate: &Value, field: &IndexAttr) -> TypeHandle {
    let aggregate_ty = aggregate.get_type(ctx).deref(ctx);
    let aggregate_ty = type_cast::<dyn AggregateType>(&*aggregate_ty).expect("Should be aggregate");
    aggregate_ty.field_ty(ctx, field.0)
}

pure_binop!("cube.bool_and", BoolAndOp);
const_eval!(BoolAndOp, {
    BoolAttr: |lhs, rhs| lhs && rhs
});

pure_binop!("cube.bool_or", BoolOrOp);
const_eval!(BoolOrOp, {
    BoolAttr: |lhs, rhs| lhs || rhs
});

pure_unop!("cube.bool_not", BoolNotOp);
const_eval!(BoolNotOp, {
    BoolAttr: |inp| !inp
});

#[cube_op(name = "cube.cast")]
#[result_ty(argument)]
#[op_interfaces(Pure)]
pub struct CastOp {
    pub input: Value,
}
erasable!(CastOp);
// TODO const_eval

#[cube_op(name = "cube.reinterpret_cast")]
#[result_ty(argument)]
#[op_interfaces(Pure)]
pub struct ReinterpretCastOp {
    pub value: Value,
}
erasable!(ReinterpretCastOp);
// TODO const_eval

#[cube_op(name = "cube.select")]
#[result_ty(same_as = true_value)]
#[op_interfaces(Pure)]
pub struct SelectOp {
    pub condition: Value,
    pub true_value: Value,
    pub false_value: Value,
}
erasable!(SelectOp);
// TODO const_eval

#[pliron_attr(name = "cube.builtin", format, verifier = "succ")]
#[derive(new, From, PartialEq, Clone, Debug)]
pub struct BuiltinAttr(pub Builtin);

#[cube_op(
    name = "cube.read_builtin",
    format = "attr($builtin, $BuiltinAttr) ` : ` type($0)"
)]
#[result_ty(argument)]
#[op_interfaces(Pure)]
pub struct ReadBuiltinOp {
    pub builtin: BuiltinAttr,
}
erasable!(ReadBuiltinOp);

#[cube_op(name = "cube.read_scalar")]
#[result_ty(from_inputs = |ctx, ty: &TypeAttr, _| ty.get_type(ctx))]
#[op_interfaces(Pure)]
pub struct ReadScalarOp {
    pub ty: TypeAttr,
    pub id: IndexAttr,
}
erasable!(ReadScalarOp);

#[cube_op(name = "cube.free")]
#[result_ty(none)]
pub struct FreeOp {
    pub memory: Value,
}

#[cube_op(name = "cube.read")]
#[result_ty(none)]
#[op_interfaces(Pure)]
pub struct DummyReadOp {
    pub value: Value,
}
erasable!(DummyReadOp);

#[cube_op(name = "cube.buffer_len")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_interfaces(Pure)]
pub struct BufferLenOp {
    pub buffer_idx: IndexAttr,
}
erasable!(BufferLenOp);

#[cube_op(name = "cube.shape")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_interfaces(Pure)]
pub struct ShapeOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}
erasable!(ShapeOp);

#[cube_op(name = "cube.stride")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_interfaces(Pure)]
pub struct StrideOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}
erasable!(StrideOp);

#[cube_op(name = "cube.comment")]
#[result_ty(none)]
pub struct CommentOp {
    pub comment: StringAttr,
}

#[pliron_op(name = "cube.printf", format, attributes = (format_string: StringAttr), verifier = "succ")]
pub struct PrintfOp;

impl PrintfOp {
    pub fn new(ctx: &mut Context, format_string: String, values: Vec<Value>) -> Self {
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        };
        op.set_attr_format_string(ctx, StringAttr::new(format_string));
        op
    }

    pub fn format_string<'a>(&self, ctx: &'a Context) -> Ref<'a, StringAttr> {
        self.get_attr_format_string(ctx).unwrap()
    }

    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}
