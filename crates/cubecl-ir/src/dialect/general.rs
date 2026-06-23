use core::cell::Ref;

use alloc::string::String;

use cubecl_macros_internal::{const_eval, cube_op, simplify};
use derive_more::From;
use derive_new::new;
use pliron::{
    builtin::attributes::{StringAttr, TypeAttr},
    derive::pliron_attr,
    r#type::{TypeHandle, type_cast},
};

use crate::{
    Builtin, ConstantValue,
    attributes::{BoolAttr, IndexAttr},
    dialect::{
        math::{int_attr, uint_attr},
        pure_binop, pure_unop,
    },
    interfaces::{AggregateType, Pure, TypedExt, erasable},
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
    BoolAttr: |lhs, rhs| lhs && rhs,
    // false && x -> false
    custom: |lhs, _| match lhs?.as_const_val() {
        ConstantValue::Bool(false) => Some(BoolAttr::new(false)),
        _ => None
    },
    // x && false -> false
    custom: |_, rhs| match rhs?.as_const_val() {
        ConstantValue::Bool(false) => Some(BoolAttr::new(false)),
        _ => None
    }
});
simplify!(BoolAndOp, {
    // true && x -> x
    |lhs, _| match lhs?.as_const_val() {
        ConstantValue::Bool(true) => Some(self.rhs(ctx)),
        _ => None,
    },
    // x && true -> x
    |_, rhs| match rhs?.as_const_val() {
        ConstantValue::Bool(true) => Some(self.lhs(ctx)),
        _ => None,
    },
    // x && x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("cube.bool_or", BoolOrOp);
const_eval!(BoolOrOp, {
    BoolAttr: |lhs, rhs| lhs || rhs,
    // true || x -> true
    custom: |lhs, _| match lhs?.as_const_val() {
        ConstantValue::Bool(true) => Some(BoolAttr::new(true)),
        _ => None
    },
    // x || true -> true
    custom: |_, rhs| match rhs?.as_const_val() {
        ConstantValue::Bool(true) => Some(BoolAttr::new(true)),
        _ => None
    }
});
simplify!(BoolOrOp, {
    // false || x -> x
    |lhs, _| match lhs?.as_const_val() {
        ConstantValue::Bool(false) => Some(self.rhs(ctx)),
        _ => None,
    },
    // false || x -> x
    |_, rhs| match rhs?.as_const_val() {
        ConstantValue::Bool(false) => Some(self.lhs(ctx)),
        _ => None,
    },
    // x || x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
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
const_eval!(ReinterpretCastOp, {
    custom: |inp| {
        let val = match inp?.as_const_val() {
            ConstantValue::Int(val) => val as u64,
            ConstantValue::Float(val) => val.to_bits(),
            ConstantValue::UInt(val) => val,
            ConstantValue::Bool(_) => None?,
        };
        let out_ty = self.get_result(ctx).get_type(ctx);
        if out_ty.is_int(ctx) {
            Some(int_attr(ctx, out_ty, val as i64))
        } else if out_ty.is_uint(ctx) {
            Some(uint_attr(ctx, out_ty, val))
        } else { // Too much weirdness around floats, don't risk it
            None
        }
    }
});

#[cube_op(name = "cube.select")]
#[result_ty(same_as = true_value)]
#[op_interfaces(Pure)]
pub struct SelectOp {
    pub condition: Value,
    pub true_value: Value,
    pub false_value: Value,
}
erasable!(SelectOp);
simplify!(SelectOp, {
    |cond, _, _| match cond?.as_const_val() {
        ConstantValue::Bool(true) => Some(self.true_value(ctx)),
        ConstantValue::Bool(false) => Some(self.false_value(ctx)),
        _ => None,
    },
    // select(cond, x, x) -> x
    |_, _, _| match self.true_value(ctx) == self.false_value(ctx) {
        true => Some(self.true_value(ctx)),
        false => None
    }
});

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
