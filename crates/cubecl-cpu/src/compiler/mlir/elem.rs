use cubecl_core::Feature;
use cubecl_core::ir;
use cubecl_core::ir::Elem;
use cubecl_core::ir::FloatKind;
use cubecl_runtime::DeviceProperties;
use melior::Context;
use melior::ir::Type;

pub(super) trait VisitType {
    fn visit<'a>(&self, context: &'a Context) -> Type<'a>;
}

impl VisitType for Elem {
    fn visit<'a>(&self, context: &'a Context) -> Type<'a> {
        match self {
            Elem::Float(FloatKind::F32) => Type::float32(context),
            _ => todo!("This type is not implemented yet."),
        }
    }
}

pub fn register_supported_types(props: &mut DeviceProperties<Feature>) {
    let supported_types = [
        // ir::Elem::UInt(ir::UIntKind::U8),
        // ir::Elem::UInt(ir::UIntKind::U16),
        // ir::Elem::UInt(ir::UIntKind::U32),
        // ir::Elem::UInt(ir::UIntKind::U64),
        // ir::Elem::Int(ir::IntKind::I8),
        // ir::Elem::Int(ir::IntKind::I16),
        // ir::Elem::Int(ir::IntKind::I32),
        // ir::Elem::Int(ir::IntKind::I64),
        // ir::Elem::AtomicInt(ir::IntKind::I32),
        // ir::Elem::AtomicInt(ir::IntKind::I64),
        // ir::Elem::AtomicUInt(ir::UIntKind::U32),
        // ir::Elem::AtomicUInt(ir::UIntKind::U64),
        // ir::Elem::Float(ir::FloatKind::BF16),
        // ir::Elem::Float(ir::FloatKind::F16),
        ir::Elem::Float(ir::FloatKind::F32),
        // ir::Elem::Float(ir::FloatKind::F64),
        // ir::Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}
