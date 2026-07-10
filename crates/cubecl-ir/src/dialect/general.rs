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
    Builtin, CanMaterialize, ConstantValue, Pure,
    attributes::{BoolAttr, IndexAttr},
    dialect::{
        math::{index_attr, int_attr},
        pure_binop, pure_unop,
    },
    interfaces::{AggregateType, ScalarType, TypedExt, aliasing::AliasingOp},
    prelude::*,
    try_cast_ty,
    types::scalar::IndexType,
};

#[cube_op(name = "cube.copy")]
#[result_ty(same_as = value)]
#[op_traits(Pure, CanMaterialize)]
pub struct CopyOp {
    pub value: Value,
}

#[op_interface_impl]
impl AliasingOp for CopyOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        if self.get_result(ctx).is_ptr(ctx) {
            Some(self.value(ctx))
        } else {
            None
        }
    }
}

#[cube_op(name = "cube.poison")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct PoisonOp {}

#[pliron_op(name = "aggregate.construct", format, verifier = "succ")]
#[op_interfaces(NResultsInterface<1>, OneResultInterface)]
#[op_traits(Pure, CanMaterialize)]
pub struct AggregateConstructOp;

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

#[op_interface_impl]
impl AliasingOp for AggregateExtractOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        let aggregate = self.aggregate(ctx);
        let field = self.field(ctx).0;
        let aggregate_ty = aggregate.get_type(ctx).deref(ctx);
        let aggregate_ty = try_cast_ty!(aggregate_ty, ctx, dyn AggregateType);
        if aggregate_ty.field_ty(ctx, field).is_ptr(ctx) {
            let construct = aggregate.defining_op().expect("Should be construct");
            Some(construct.operand(ctx, field))
        } else {
            None
        }
    }
}

#[cube_op(name = "aggregate.extract")]
#[result_ty(from_inputs = aggregate_extract_type)]
#[op_traits(Pure, CanMaterialize)]
pub struct AggregateExtractOp {
    pub aggregate: Value,
    pub field: IndexAttr,
}

fn aggregate_extract_type(ctx: &Context, aggregate: &Value, field: &IndexAttr) -> TypeHandle {
    let aggregate_ty = aggregate.get_type(ctx).deref(ctx);
    let aggregate_ty = type_cast::<dyn AggregateType>(&*aggregate_ty).expect("Should be aggregate");
    aggregate_ty.field_ty(ctx, field.0)
}

pure_binop!("cube.bool_and", BoolAndOp);
const_eval!(BoolAndOp, {
    BoolAttr: |lhs, rhs| lhs && rhs,
    // false && x -> false
    custom: |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => Some(BoolAttr::new(false)),
        _ => None
    },
    // x && false -> false
    custom: |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => Some(BoolAttr::new(false)),
        _ => None
    }
});
simplify!(BoolAndOp, {
    // true && x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(self.rhs(ctx)),
        _ => None,
    },
    // x && true -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
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
    custom: |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(BoolAttr::new(true)),
        _ => None
    },
    // x || true -> true
    custom: |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(BoolAttr::new(true)),
        _ => None
    }
});
simplify!(BoolOrOp, {
    // false || x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => Some(self.rhs(ctx)),
        _ => None,
    },
    // false || x -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
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
#[op_traits(Pure, CanMaterialize)]
pub struct CastOp {
    pub input: Value,
}
const_eval!(CastOp, {
    custom: |inp| {
        let val = inp?.as_const_val(ctx);
        let out_ty = self.get_result(ctx).get_type(ctx).deref(ctx);
        let elem = type_cast::<dyn ScalarType>(&*out_ty)?.elem_type(ctx);
        Some(val.cast_to(elem).as_attribute(ctx, elem))
    }
});

#[cube_op(name = "cube.reinterpret_cast")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct ReinterpretCastOp {
    pub input: Value,
}
const_eval!(ReinterpretCastOp, {
    custom: |inp| {
        // Too much weirdness around floats, don't bother dealing with it
        let val = match inp?.as_const_val(ctx) {
            ConstantValue::Int(val) => val as u64,
            ConstantValue::UInt(val) => val,
            _ => None?,
        };
        let out_ty = self.get_result(ctx).get_type(ctx);
        if out_ty.is_int(ctx) {
            Some(int_attr(ctx, out_ty, val as i128))
        } else if out_ty.is_index(ctx) {
            Some(index_attr(val as usize))
        } else {
            None
        }
    }
});

#[cube_op(name = "cube.select")]
#[result_ty(same_as = true_value)]
#[op_traits(Pure, CanMaterialize)]
pub struct SelectOp {
    pub condition: Value,
    pub true_value: Value,
    pub false_value: Value,
}
simplify!(SelectOp, {
    |cond, _, _| match cond?.as_const_val(ctx) {
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
#[op_traits(Pure, CanMaterialize)]
pub struct ReadBuiltinOp {
    pub builtin: BuiltinAttr,
}

#[cube_op(name = "cube.read_scalar")]
#[result_ty(from_inputs = |ctx, ty: &TypeAttr, _| ty.get_type(ctx))]
#[op_traits(Pure, CanMaterialize)]
pub struct ReadScalarOp {
    pub ty: TypeAttr,
    pub id: IndexAttr,
}

#[cube_op(name = "cube.free")]
#[result_ty(none)]
pub struct FreeOp {
    pub memory: Value,
}

#[cube_op(name = "cube.buffer_len")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct BufferLenOp {
    pub buffer_idx: IndexAttr,
}

#[cube_op(name = "cube.shape")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct ShapeOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}

#[cube_op(name = "cube.stride")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct StrideOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}

#[cube_op(name = "cube.comment")]
#[result_ty(none)]
pub struct CommentOp {
    pub comment: StringAttr,
}

#[pliron_op(name = "cube.printf", format, attributes = (cube_printf_format_string: StringAttr), verifier = "succ")]
pub struct PrintfOp;

impl PrintfOp {
    pub fn new(ctx: &mut Context, format_string: String, values: Vec<Value>) -> Self {
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        };
        op.set_attr_cube_printf_format_string(ctx, StringAttr::new(format_string));
        op
    }

    pub fn format_string<'a>(&self, ctx: &'a Context) -> Ref<'a, StringAttr> {
        self.get_attr_cube_printf_format_string(ctx).unwrap()
    }

    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}
