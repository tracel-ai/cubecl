use cubecl_core::ir::{self as cube, ConstantScalarValue, FloatKind, Id, UIntKind};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    GlobalInputArray(Id, Item),
    GlobalOutputArray(Id, Item),
    GlobalScalar(Id, Elem, cube::Elem),
    ConstantScalar(ConstantScalarValue, Elem),
    LocalMut {
        id: Id,
        item: Item,
    },
    LocalConst {
        id: Id,
        item: Item,
    },
    Named {
        name: String,
        item: Item,
        is_array: bool,
    },
    // TODO: Potential cleanup, seems that variable is not used at all
    LocalScalar {
        id: Id,
        elem: Elem,
    },
    SharedMemory(Id, Item, u32),
    ConstantArray(Id, Item, u32),
    LocalArray(Id, Item, u32),
    Id,
    LocalInvocationIndex,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    WorkgroupSize,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroups,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    SubgroupSize,
    SubgroupInvocationId,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F16,
    F32,
    F64,
    AtomicF32,
    I32,
    I64,
    AtomicI32,
    U32,
    U64,
    AtomicU32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Variable {
    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::GlobalScalar(_, _, _) => true,
            Variable::ConstantScalar(_, _) => true,
            Variable::LocalScalar { .. } => true,
            Variable::Id => true,
            Variable::LocalInvocationIndex => true,
            Variable::LocalInvocationIdX => true,
            Variable::LocalInvocationIdY => true,
            Variable::LocalInvocationIdZ => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::LocalArray(_, _, _) => false,
            Variable::LocalMut { .. } => false,
            Variable::LocalConst { .. } => false,
            Variable::Named { .. } => false,
            Variable::WorkgroupIdX => true,
            Variable::WorkgroupIdY => true,
            Variable::WorkgroupIdZ => true,
            Variable::GlobalInvocationIdX => true,
            Variable::GlobalInvocationIdY => true,
            Variable::GlobalInvocationIdZ => true,
            Variable::WorkgroupSizeX => true,
            Variable::WorkgroupSizeY => true,
            Variable::WorkgroupSizeZ => true,
            Variable::NumWorkgroupsX => true,
            Variable::NumWorkgroupsY => true,
            Variable::NumWorkgroupsZ => true,
            Variable::WorkgroupId => true,
            Variable::WorkgroupSize => true,
            Variable::NumWorkgroups => true,
            Variable::SubgroupSize => true,
            Variable::SubgroupInvocationId => true,
        }
    }
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }
    pub fn is_atomic(&self) -> bool {
        match self {
            Variable::GlobalInputArray(_, item) => item.elem().is_atomic(),
            Variable::GlobalOutputArray(_, item) => item.elem().is_atomic(),
            Variable::GlobalScalar(_, elem, _) => elem.is_atomic(),
            Variable::LocalMut { item, .. } => item.elem().is_atomic(),
            Variable::Named { item, .. } => item.elem().is_atomic(),
            Variable::LocalScalar { elem, .. } => elem.is_atomic(),
            Variable::SharedMemory(_, item, _) => item.elem().is_atomic(),
            Variable::LocalArray(_, item, _) => item.elem().is_atomic(),
            _ => false,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Self::GlobalInputArray(_, e) => *e,
            Self::GlobalOutputArray(_, e) => *e,
            Self::SharedMemory(_, e, _) => *e,
            Self::ConstantArray(_, e, _) => *e,
            Self::LocalArray(_, e, _) => *e,
            Self::LocalMut { item, .. } => *item,
            Self::LocalConst { item, .. } => *item,
            Self::Named { item, .. } => *item,
            Self::ConstantScalar(_, e) => Item::Scalar(*e),
            Self::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Self::Id => Item::Scalar(Elem::U32),
            Self::LocalInvocationIndex => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdX => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdY => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdZ => Item::Scalar(Elem::U32),
            Self::LocalScalar { elem, .. } => Item::Scalar(*elem),
            Self::WorkgroupId => Item::Scalar(Elem::U32),
            Self::WorkgroupIdX => Item::Scalar(Elem::U32),
            Self::WorkgroupIdY => Item::Scalar(Elem::U32),
            Self::WorkgroupIdZ => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdX => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdY => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdZ => Item::Scalar(Elem::U32),
            Self::WorkgroupSize => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeX => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeY => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeZ => Item::Scalar(Elem::U32),
            Self::NumWorkgroups => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsX => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsY => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsZ => Item::Scalar(Elem::U32),
            Self::SubgroupSize => Item::Scalar(Elem::U32),
            Self::SubgroupInvocationId => Item::Scalar(Elem::U32),
        }
    }
    pub fn elem(&self) -> Elem {
        *self.item().elem()
    }

    pub fn fmt_cast_to(&self, item: Item) -> String {
        if self.item() != item {
            format!("{item}({self})")
        } else {
            format!("{self}")
        }
    }
}

impl Item {
    pub fn elem(&self) -> &Elem {
        match self {
            Item::Vec4(e) => e,
            Item::Vec3(e) => e,
            Item::Vec2(e) => e,
            Item::Scalar(e) => e,
        }
    }

    pub fn vectorization_factor(&self) -> usize {
        match self {
            Item::Vec4(_) => 4,
            Item::Vec3(_) => 3,
            Item::Vec2(_) => 2,
            Item::Scalar(_) => 1,
        }
    }

    pub fn fmt_cast_to(&self, item: Item, text: String) -> String {
        if *self != item {
            format!("{item}({text})")
        } else {
            text
        }
    }
}

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Self::F16 => core::mem::size_of::<half::f16>(),
            Self::F32 => core::mem::size_of::<f32>(),
            Self::F64 => core::mem::size_of::<f64>(),
            Self::AtomicF32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::I64 => core::mem::size_of::<i64>(),
            Self::AtomicI32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::U64 => core::mem::size_of::<u64>(),
            Self::AtomicU32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, Self::AtomicI32 | Self::AtomicU32 | Self::AtomicF32)
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => f.write_str("f16"),
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
            Self::AtomicF32 => f.write_str("atomic<f32>"),
            Self::I32 => f.write_str("i32"),
            Self::I64 => f.write_str("i64"),
            Self::AtomicI32 => f.write_str("atomic<i32>"),
            Self::U32 => f.write_str("u32"),
            Self::U64 => f.write_str("u64"),
            Self::AtomicU32 => f.write_str("atomic<u32>"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Vec4(elem) => write!(f, "vec4<{elem}>"),
            Item::Vec3(elem) => write!(f, "vec3<{elem}>"),
            Item::Vec2(elem) => write!(f, "vec2<{elem}>"),
            Item::Scalar(elem) => write!(f, "{elem}"),
        }
    }
}

fn format_number(num: f64, suffix: &str) -> String {
    let formatted = format!("{:.34}", num);
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string() + suffix
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(number, _) => {
                write!(f, "buffer_{number}_global")
            }
            Variable::LocalScalar { id: index, .. } => write!(f, "s_{index}"),
            Variable::LocalMut { id, .. } => write!(f, "l_mut_{id}"),
            Variable::LocalConst { id, .. } => write!(f, "l_{id}"),
            Variable::Named { name, .. } => f.write_str(name),
            Variable::GlobalOutputArray(number, _) => {
                write!(f, "buffer_{number}_global")
            }
            Variable::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Variable::ConstantScalar(number, _elem) => match number {
                ConstantScalarValue::Int(val, _) => write!(f, "{}", *val),
                ConstantScalarValue::Float(val, kind) => match kind {
                    FloatKind::BF16 | FloatKind::TF32 => {
                        todo!("Unsupported")
                    }
                    FloatKind::F16 => f.write_str(&format_number(*val, "h")),
                    FloatKind::F32 | FloatKind::Flex32 => f.write_str(&format_number(*val, "f")),
                    FloatKind::F64 => f.write_str(&format_number(*val, "lf")),
                },
                ConstantScalarValue::UInt(val, UIntKind::U32) => write!(f, "{}u", *val),
                ConstantScalarValue::UInt(val, UIntKind::U64) => write!(f, "{}lu", *val),
                ConstantScalarValue::UInt(_, _) => unimplemented!("Unsupported"),
                ConstantScalarValue::Bool(val) => write!(f, "{}", val),
            },
            Variable::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }
            Variable::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
            Variable::LocalArray(number, _, _) => {
                write!(f, "a_{number}")
            }
            Variable::Id => f.write_str("id"),
            Variable::LocalInvocationIndex => f.write_str("local_idx"),
            Variable::LocalInvocationIdX => f.write_str("local_invocation_id.x"),
            Variable::LocalInvocationIdY => f.write_str("local_invocation_id.y"),
            Variable::LocalInvocationIdZ => f.write_str("local_invocation_id.z"),
            Variable::WorkgroupId => f.write_str("workgroup_id_no_axis"),
            Variable::WorkgroupIdX => f.write_str("workgroup_id.x"),
            Variable::WorkgroupIdY => f.write_str("workgroup_id.y"),
            Variable::WorkgroupIdZ => f.write_str("workgroup_id.z"),
            Variable::GlobalInvocationIdX => f.write_str("global_id.x"),
            Variable::GlobalInvocationIdY => f.write_str("global_id.y"),
            Variable::GlobalInvocationIdZ => f.write_str("global_id.z"),
            Variable::WorkgroupSizeX => f.write_str("WORKGROUP_SIZE_X"),
            Variable::WorkgroupSizeY => f.write_str("WORKGROUP_SIZE_Y"),
            Variable::WorkgroupSizeZ => f.write_str("WORKGROUP_SIZE_Z"),
            Variable::NumWorkgroupsX => f.write_str("num_workgroups.x"),
            Variable::NumWorkgroupsY => f.write_str("num_workgroups.y"),
            Variable::NumWorkgroupsZ => f.write_str("num_workgroups.z"),
            Variable::WorkgroupSize => f.write_str("workgroup_size_no_axis"),
            Variable::NumWorkgroups => f.write_str("num_workgroups_no_axis"),
            Variable::SubgroupSize => f.write_str("subgroup_size"),
            Variable::SubgroupInvocationId => f.write_str("subgroup_invocation_id"),
        }
    }
}

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let item = var.item();
        let index = self.index;

        match &self.var {
            Variable::GlobalScalar(_, _, _) => write!(f, "{var}"),
            var if matches!(item, Item::Scalar(_)) => write!(f, "{var}"),
            var => write!(f, "{var}[{index}]"),
        }
    }
}

impl Variable {
    pub fn fmt_left(&self) -> String {
        match self {
            Variable::LocalConst { .. } => {
                format!("let {self}")
            }
            var => format!("{}", var),
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, Variable::LocalConst { .. })
    }
}

impl IndexedVariable {
    pub fn fmt_left(&self) -> String {
        let item = self.var.item();
        match &self.var {
            Variable::GlobalScalar(_, _, _) => self.var.fmt_left(),
            var if matches!(item, Item::Scalar(_)) => var.fmt_left(),
            _ => format!("{self}"),
        }
    }

    pub fn fmt_cast(&self, item: Item) -> String {
        if self.var.item() != item {
            format!("{item}({self})")
        } else {
            format!("{self}")
        }
    }
}
