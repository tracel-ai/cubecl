use cubecl_core::Feature;
use cubecl_core::ir;
use cubecl_core::ir::Elem;
use cubecl_core::ir::FloatKind;
use cubecl_core::ir::IntKind;
use cubecl_core::ir::UIntKind;
use cubecl_runtime::DeviceProperties;
use melior::dialect::index;
use melior::ir::Type;
use melior::ir::Value;
use melior::ir::ValueLike;
use melior::ir::r#type::IntegerType;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn elem_to_type(&self, elem: Elem) -> Type<'a> {
        match elem {
            Elem::Float(FloatKind::BF16) => Type::bfloat16(self.context),
            Elem::Float(FloatKind::F16) => Type::float16(self.context),
            Elem::Float(FloatKind::F32) => Type::float32(self.context),
            Elem::Float(FloatKind::F64) => Type::float64(self.context),
            Elem::Int(IntKind::I8) => IntegerType::new(self.context, 8).into(),
            Elem::Int(IntKind::I16) => IntegerType::new(self.context, 16).into(),
            Elem::Int(IntKind::I32) => IntegerType::new(self.context, 32).into(),
            Elem::Int(IntKind::I64) => IntegerType::new(self.context, 64).into(),
            Elem::UInt(UIntKind::U8) => IntegerType::new(self.context, 8).into(),
            Elem::UInt(UIntKind::U16) => IntegerType::new(self.context, 16).into(),
            Elem::UInt(UIntKind::U32) => IntegerType::new(self.context, 32).into(),
            Elem::UInt(UIntKind::U64) => IntegerType::new(self.context, 64).into(),
            Elem::Bool => IntegerType::new(self.context, 1).into(),
            _ => todo!("This type is not implemented yet."),
        }
    }

    pub fn visit_correct_index(
        &self,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        if lhs.r#type() == Type::index(self.context) && rhs.r#type() != Type::index(self.context) {
            let rhs = self.append_operation_with_result(index::casts(
                rhs,
                Type::index(self.context),
                self.location,
            ));
            (lhs, rhs)
        } else if lhs.r#type() != Type::index(self.context)
            && rhs.r#type() == Type::index(self.context)
        {
            let lhs = self.append_operation_with_result(index::casts(
                lhs,
                Type::index(self.context),
                self.location,
            ));
            (lhs, rhs)
        } else {
            (lhs, rhs)
        }
    }
}

pub fn register_supported_types(props: &mut DeviceProperties<Feature>) {
    let supported_types = [
        Elem::UInt(ir::UIntKind::U8),
        Elem::UInt(ir::UIntKind::U16),
        Elem::UInt(ir::UIntKind::U32),
        Elem::UInt(ir::UIntKind::U64),
        Elem::Int(ir::IntKind::I8),
        Elem::Int(ir::IntKind::I16),
        Elem::Int(ir::IntKind::I32),
        Elem::Int(ir::IntKind::I64),
        // Elem::AtomicInt(ir::IntKind::I32),
        // Elem::AtomicInt(ir::IntKind::I64),
        // Elem::AtomicUInt(ir::UIntKind::U32),
        // Elem::AtomicUInt(ir::UIntKind::U64),
        Elem::Float(ir::FloatKind::BF16),
        Elem::Float(ir::FloatKind::F16),
        Elem::Float(ir::FloatKind::F32),
        Elem::Float(ir::FloatKind::F64),
        // Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}
