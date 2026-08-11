use alloc::{
    boxed::Box,
    string::{String, ToString},
};
use cubecl_macros_internal::{cube_op, op_traits};
use pliron::{
    attribute::AttrObj,
    printable::Printable,
    r#type::TypeHandle,
    utils::table::{HMap, SmallSet},
    verify_err,
};
use thiserror::Error;

use crate::{
    CanMaterialize, Pure,
    attributes::IndexAttr,
    interfaces::{
        aliasing::AliasingOp,
        memory_slot::{
            DeletionKind, DestructurableAccessorOpInterface, DestructurableConstructorOpInterface,
            DestructurableTypeInterface, DestructurableValueSlot, ValueSlot,
        },
        *,
    },
    prelude::*,
    try_cast_ty,
    types::{VectorType, aggregate::index_attr, scalar::IndexType},
};

#[pliron_op(
    name = "composite.construct",
    format = "operands(CharSpace(`,`)) ` : ` type($0)"
)]
#[op_interfaces(NResultsInterface<1>, OneResultInterface, AtLeastNOpdsInterface<1>)]
#[op_traits(Pure, CanMaterialize)]
pub struct CompositeConstructOp;

impl CompositeConstructOp {
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
impl DestructurableConstructorOpInterface for CompositeConstructOp {
    fn destructurable_values(&self, ctx: &Context) -> Vec<DestructurableValueSlot> {
        let ty = self.result_type(ctx).deref(ctx);
        let Some(destructurable) = type_cast::<dyn DestructurableTypeInterface>(&*ty) else {
            return vec![];
        };
        let Some(subelement_types) = destructurable.subelement_index_map(ctx) else {
            return vec![];
        };
        vec![DestructurableValueSlot {
            slot: ValueSlot::new(self.get_result(ctx), self.result_type(ctx)),
            subelement_types,
        }]
    }

    fn destructure(
        &self,
        ctx: &mut Context,
        _value: &DestructurableValueSlot,
        used_indices: &SmallSet<AttrObj, 8>,
        _rewriter: &mut PassRewriter,
        _new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
    ) -> HMap<AttrObj, ValueSlot> {
        let op = self.get_operation();
        let mut slot_map = HMap::new();
        for used_index in used_indices {
            let index = used_index.downcast_ref::<IndexAttr>().unwrap().0;
            let opd = op.operand(ctx, index);
            slot_map.insert(used_index.clone(), ValueSlot::new(opd, opd.get_type(ctx)));
        }
        slot_map
    }

    fn handle_destructuring_complete(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        rewriter: &mut PassRewriter,
    ) -> Option<TraitOp<dyn DestructurableConstructorOpInterface>> {
        assert_eq!(value.slot.value, self.get_result(ctx));
        rewriter.erase_operation(ctx, self.get_operation());
        None
    }
}

#[cube_op(
    name = "composite.extract",
    format = "$0 `[` attr($index, $IndexAttr) `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(from_inputs = composite_extract_type)]
#[op_traits(Pure, CanMaterialize)]
pub struct CompositeExtractOp {
    pub composite: Value,
    pub index: IndexAttr,
}

fn composite_extract_type(ctx: &Context, aggregate: &Value, field: &IndexAttr) -> TypeHandle {
    let aggregate_ty = aggregate.get_type(ctx).deref(ctx);
    let aggregate_ty =
        type_cast::<dyn DestructurableTypeInterface>(&*aggregate_ty).expect("Should be aggregate");
    aggregate_ty.type_at_index(ctx, &index_attr(field.0))
}

#[derive(Error, Debug)]
pub enum CompositeConstructError {
    #[error(
        "[CompositeConstructOp]: Output composite size doesn't match parameter count: Expected {_0} parameters, got {_1}"
    )]
    ParameterCountMismatch(usize, usize),
    #[error(
        "[CompositeConstructOp]: Output field type doesn't match parameter type: Expected {_0}, got {_1}"
    )]
    ParameterTypeMismatch(String, String),
}

#[op_interface_impl]
impl AliasingOp for CompositeExtractOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        let aggregate = self.composite(ctx);
        let field = self.index(ctx).0;
        let aggregate_ty = aggregate.get_type(ctx).deref(ctx);
        let destruct = try_cast_ty!(aggregate_ty, ctx, dyn DestructurableTypeInterface);
        if destruct.type_at_index(ctx, &index_attr(field)).is_ptr(ctx) {
            let construct = aggregate.defining_op().expect("Should be construct");
            Some(construct.operand(ctx, field))
        } else {
            None
        }
    }
}

impl Verify for CompositeConstructOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ty = self.result_type(ctx).deref(ctx);
        let opds = self.get_operation().operands(ctx);
        let destructurable = try_cast_ty!(ty, ctx, dyn DestructurableTypeInterface);
        let fields = destructurable.subelement_index_map(ctx).unwrap();

        if opds.len() != fields.len() {
            return verify_err!(
                self.loc(ctx),
                CompositeConstructError::ParameterCountMismatch(fields.len(), opds.len())
            );
        }

        for (i, &opd) in opds.iter().enumerate() {
            let field_ty = fields[&index_attr(i)];
            if opd.get_type(ctx) != field_ty {
                return verify_err!(
                    self.loc(ctx),
                    CompositeConstructError::ParameterTypeMismatch(
                        field_ty.disp(ctx).to_string(),
                        opd.get_type(ctx).disp(ctx).to_string()
                    )
                );
            }
        }

        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum CompositeOpError {
    #[error("[CompositeOp]: Index is out of range: index is {_0} but vectorization is {_1}.")]
    IndexOutOfRange(usize, usize),
    #[error(
        "[CompositeOp]: Field type doesn't match the inner type of the composite: expected {_0}, got {_1}"
    )]
    MismatchedFieldType(String, String),
}

impl Verify for CompositeExtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ty = self.composite(ctx).get_type(ctx).deref(ctx);
        let destructurable = try_cast_ty!(ty, ctx, dyn DestructurableTypeInterface);
        let fields = destructurable.subelement_index_map(ctx).unwrap();

        let loc = self.loc(ctx);
        let index = self.index(ctx).0;
        if index >= fields.len() {
            return verify_err!(loc, CompositeOpError::IndexOutOfRange(index, fields.len()));
        }
        Ok(())
    }
}

#[op_interface_impl]
impl DestructurableAccessorOpInterface for CompositeExtractOp {
    fn can_rewire(
        &self,
        ctx: &Context,
        value: &DestructurableValueSlot,
        used_indices: &mut SmallSet<AttrObj, 8>,
        _must_be_safely_used: &mut Vec<ValueSlot>,
    ) -> bool {
        if value.slot.value != self.composite(ctx) {
            return false;
        }
        used_indices.insert(Box::new(*self.index(ctx)));
        true
    }

    fn rewire(
        &self,
        ctx: &mut Context,
        _value: &DestructurableValueSlot,
        subvalues: &HMap<AttrObj, ValueSlot>,
        rewriter: &mut PassRewriter,
    ) -> DeletionKind {
        let index: AttrObj = Box::new(*self.index(ctx));
        let slot = &subvalues[&index];
        rewriter.replace_value_uses_with(ctx, self.get_result(ctx), slot.value);
        DeletionKind::Delete
    }
}

#[cube_op(
    name = "composite.insert",
    format = "$1 ` -> ` $0 `[` attr($index, $IndexAttr) `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(same_as = composite)]
#[op_interfaces(OperandNOfType<0, VectorType>, ResultNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct CompositeInsertOp {
    pub composite: Value,
    pub value: Value,
    pub index: IndexAttr,
}

impl Verify for CompositeInsertOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ty = self.result_type(ctx).deref(ctx);
        let destructurable = try_cast_ty!(ty, ctx, dyn DestructurableTypeInterface);
        let fields = destructurable.subelement_index_map(ctx).unwrap();

        let loc = self.loc(ctx);
        let index = self.index(ctx).0;
        if index >= fields.len() {
            return verify_err!(loc, CompositeOpError::IndexOutOfRange(index, fields.len()));
        }
        let field_ty = fields[&index_attr(index)];
        let value_ty = self.value(ctx).get_type(ctx);
        if field_ty != value_ty {
            return verify_err!(
                loc,
                CompositeOpError::MismatchedFieldType(
                    field_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl DestructurableConstructorOpInterface for CompositeInsertOp {
    fn destructurable_values(&self, ctx: &Context) -> Vec<DestructurableValueSlot> {
        let ty = self.result_type(ctx).deref(ctx);
        let Some(destructurable) = type_cast::<dyn DestructurableTypeInterface>(&*ty) else {
            return vec![];
        };
        let Some(subelement_types) = destructurable.subelement_index_map(ctx) else {
            return vec![];
        };
        vec![DestructurableValueSlot {
            slot: ValueSlot::new(self.get_result(ctx), self.result_type(ctx)),
            subelement_types,
        }]
    }

    fn destructure(
        &self,
        ctx: &mut Context,
        _value: &DestructurableValueSlot,
        used_indices: &SmallSet<AttrObj, 8>,
        rewriter: &mut PassRewriter,
        _new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
    ) -> HMap<AttrObj, ValueSlot> {
        let inserted_index = self.index(ctx).0;
        let mut slot_map = HMap::new();
        for used_index in used_indices {
            let index = used_index.downcast_ref::<IndexAttr>().unwrap().0;
            let value = if index == inserted_index {
                ValueSlot::new(self.value(ctx), self.value(ctx).get_type(ctx))
            } else {
                let extract = CompositeExtractOp::new(ctx, self.composite(ctx), index);
                rewriter.append_op(ctx, &extract);
                ValueSlot::new(extract.get_result(ctx), extract.result_type(ctx))
            };
            slot_map.insert(used_index.clone(), value);
        }
        slot_map
    }

    fn handle_destructuring_complete(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        rewriter: &mut PassRewriter,
    ) -> Option<TraitOp<dyn DestructurableConstructorOpInterface>> {
        assert_eq!(value.slot.value, self.get_result(ctx));
        rewriter.erase_operation(ctx, self.get_operation());
        None
    }
}

#[cube_op(
    name = "vector.broadcast",
    format = "$0 ` : ` type($0)",
    verifier = "custom"
)]
#[result_ty(argument)]
#[op_interfaces(ResultNOfType<0, VectorType>, TriviallyUnrollable)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorBroadcastOp {
    pub input: Value,
}

impl Verify for VectorBroadcastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let value_ty = self.input(ctx).get_type(ctx);
        let scalar_ty = self.get_result(ctx).scalar_ty(ctx);
        if scalar_ty != value_ty {
            return verify_err!(
                loc,
                CompositeOpError::MismatchedFieldType(
                    scalar_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl DestructurableConstructorOpInterface for VectorBroadcastOp {
    fn destructurable_values(&self, ctx: &Context) -> Vec<DestructurableValueSlot> {
        let ty = self.result_type(ctx).deref(ctx);
        let Some(destructurable) = type_cast::<dyn DestructurableTypeInterface>(&*ty) else {
            return vec![];
        };
        let Some(subelement_types) = destructurable.subelement_index_map(ctx) else {
            return vec![];
        };
        vec![DestructurableValueSlot {
            slot: ValueSlot::new(self.get_result(ctx), self.result_type(ctx)),
            subelement_types,
        }]
    }

    fn destructure(
        &self,
        ctx: &mut Context,
        _value: &DestructurableValueSlot,
        used_indices: &SmallSet<AttrObj, 8>,
        _rewriter: &mut PassRewriter,
        _new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
    ) -> HMap<AttrObj, ValueSlot> {
        let mut slot_map = HMap::new();
        for used_index in used_indices {
            let slot = ValueSlot::new(self.input(ctx), self.input(ctx).get_type(ctx));
            slot_map.insert(used_index.clone(), slot);
        }
        slot_map
    }

    fn handle_destructuring_complete(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        rewriter: &mut PassRewriter,
    ) -> Option<TraitOp<dyn DestructurableConstructorOpInterface>> {
        assert_eq!(value.slot.value, self.get_result(ctx));
        rewriter.erase_operation(ctx, self.get_operation());
        None
    }
}

#[cube_op(
    name = "vector.insert_dynamic",
    format = "$1 ` -> ` $0 `[` $2 `] : ` type($0)",
    verifier = "custom"
)]
#[result_ty(same_as = vector)]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<2, IndexType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorInsertDynamicOp {
    pub vector: Value,
    pub value: Value,
    pub index: Value,
}

impl Verify for VectorInsertDynamicOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let scalar_ty = self.vector(ctx).scalar_ty(ctx);
        let value_ty = self.value(ctx).get_type(ctx);
        if scalar_ty != value_ty {
            verify_err!(
                loc,
                CompositeOpError::MismatchedFieldType(
                    scalar_ty.disp(ctx).to_string(),
                    value_ty.disp(ctx).to_string()
                )
            )?;
        }
        Ok(())
    }
}

#[cube_op(name = "vector.extract_dynamic", format = "$0 `[` $1 `] : ` type($0)")]
#[result_ty(from_inputs = |ctx, vector, _| scalar_ty(ctx, vector))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, IndexType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct VectorExtractDynamicOp {
    pub vector: Value,
    pub index: Value,
}

#[cube_op(name = "vector.magnitude")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct MagnitudeOp {
    pub input: Value,
}

#[cube_op(name = "vector.normalize")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct NormalizeOp {
    pub input: Value,
}

#[cube_op(name = "vector.i_sum")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct ISumOp {
    pub input: Value,
}

#[cube_op(name = "vector.f_sum")]
#[result_ty(from_inputs = scalar_ty)]
#[op_interfaces(OperandNOfType<0, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct FSumOp {
    pub input: Value,
}

#[cube_op(name = "vector.s_dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct SDotOp {
    pub lhs: Value,
    pub rhs: Value,
}

#[cube_op(name = "vector.u_dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct UDotOp {
    pub lhs: Value,
    pub rhs: Value,
}

#[cube_op(name = "vector.f_dot")]
#[result_ty(from_inputs = |ctx, lhs, _| scalar_ty(ctx, lhs))]
#[op_interfaces(OperandNOfType<0, VectorType>, OperandNOfType<1, VectorType>)]
#[op_traits(CanMaterialize, Pure)]
pub struct FDotOp {
    pub lhs: Value,
    pub rhs: Value,
}

fn scalar_ty(ctx: &Context, input: &Value) -> TypeHandle {
    input.scalar_ty(ctx)
}
