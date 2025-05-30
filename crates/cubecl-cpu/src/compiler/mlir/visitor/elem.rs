use cubecl_core::Feature;
use cubecl_core::ir;
use cubecl_core::ir::Elem;
use cubecl_core::ir::FloatKind;
use cubecl_runtime::DeviceProperties;
use melior::dialect::index;
use melior::ir::Type;
use melior::ir::Value;
use melior::ir::ValueLike;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn elem_to_type(&self, elem: Elem) -> Type<'a> {
        match elem {
            Elem::Float(FloatKind::BF16) => Type::bfloat16(self.context),
            Elem::Float(FloatKind::F16) => Type::float16(self.context),
            Elem::Float(FloatKind::F32) => Type::float32(self.context),
            Elem::Float(FloatKind::F64) => Type::float64(self.context),
            _ => todo!("This type is not implemented yet."),
        }
    }

    pub fn is_signed_int(&self, elem: Elem) -> bool {
        matches!(elem, Elem::Int(_) | Elem::AtomicInt(_))
    }

    pub fn is_unsigned_int(&self, elem: Elem) -> bool {
        matches!(elem, Elem::UInt(_) | Elem::AtomicUInt(_))
    }

    pub fn is_float(&self, elem: Elem) -> bool {
        matches!(elem, Elem::Float(_) | Elem::AtomicFloat(_))
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
