use cubecl_core::ir::{self as cube, ConstantScalarValue, FloatKind, IntKind};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    SubgroupSize,
    GlobalInputArray(u16, Item),
    GlobalOutputArray(u16, Item),
    GlobalScalar(u16, Elem, cube::Elem),
    ConstantScalar(ConstantScalarValue, Elem),
    Local {
        id: u16,
        item: Item,
        depth: u8,
    },
    LocalBinding {
        id: u16,
        item: Item,
    },
    Named {
        name: String,
        item: Item,
        is_array: bool,
    },
    Slice {
        id: u16,
        item: Item,
        depth: u8,
    },
    LocalScalar {
        id: u16,
        elem: Elem,
        depth: u8,
    },
    SharedMemory(u16, Item, u32),
    ConstantArray(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
    Id,
    LocalInvocationIndex,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    Rank,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    I32,
    AtomicI32,
    U32,
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
            Self::GlobalScalar(_, _, _) => true,
            Self::ConstantScalar(_, _) => true,
            Self::LocalScalar { .. } => true,
            Self::Id => true,
            Self::LocalInvocationIndex => true,
            Self::LocalInvocationIdX => true,
            Self::LocalInvocationIdY => true,
            Self::LocalInvocationIdZ => true,
            Self::Rank => true,
            Self::GlobalInputArray(_, _) => false,
            Self::GlobalOutputArray(_, _) => false,
            Self::SharedMemory(_, _, _) => false,
            Self::ConstantArray(_, _, _) => false,
            Self::LocalArray(_, _, _, _) => false,
            Self::Local { .. } => false,
            Self::LocalBinding { .. } => false,
            Self::Named { .. } => false,
            Self::Slice { .. } => false,
            Self::WorkgroupIdX => true,
            Self::WorkgroupIdY => true,
            Self::WorkgroupIdZ => true,
            Self::GlobalInvocationIdX => true,
            Self::GlobalInvocationIdY => true,
            Self::GlobalInvocationIdZ => true,
            Self::WorkgroupSizeX => true,
            Self::WorkgroupSizeY => true,
            Self::WorkgroupSizeZ => true,
            Self::NumWorkgroupsX => true,
            Self::NumWorkgroupsY => true,
            Self::NumWorkgroupsZ => true,
            Self::WorkgroupId => true,
            Self::WorkgroupSize => true,
            Self::NumWorkgroups => true,
            Self::SubgroupSize => true,
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
            Self::GlobalInputArray(_, item) => item.elem().is_atomic(),
            Self::GlobalOutputArray(_, item) => item.elem().is_atomic(),
            Self::GlobalScalar(_, elem, _) => elem.is_atomic(),
            Self::Local { item, .. } => item.elem().is_atomic(),
            Self::Named { item, .. } => item.elem().is_atomic(),
            Self::Slice { item, .. } => item.elem().is_atomic(),
            Self::LocalScalar { elem, .. } => elem.is_atomic(),
            Self::SharedMemory(_, item, _) => item.elem().is_atomic(),
            Self::LocalArray(_, item, _, _) => item.elem().is_atomic(),
            _ => false,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Self::GlobalInputArray(_, e) => *e,
            Self::GlobalOutputArray(_, e) => *e,
            Self::SharedMemory(_, e, _) => *e,
            Self::ConstantArray(_, e, _) => *e,
            Self::LocalArray(_, e, _, _) => *e,
            Self::Local { item, .. } => *item,
            Self::LocalBinding { item, .. } => *item,
            Self::Slice { item, .. } => *item,
            Self::Named { item, .. } => *item,
            Self::ConstantScalar(_, e) => Item::Scalar(*e),
            Self::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Self::Id => Item::Scalar(Elem::U32),
            Self::LocalInvocationIndex => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdX => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdY => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdZ => Item::Scalar(Elem::U32),
            Self::Rank => Item::Scalar(Elem::U32),
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
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::AtomicI32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::AtomicU32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, Self::AtomicI32 | Self::AtomicU32)
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::I32 => f.write_str("i32"),
            Self::AtomicI32 => f.write_str("atomic<i32>"),
            Self::U32 => f.write_str("u32"),
            Self::AtomicU32 => f.write_str("atomic<u32>"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vec4(elem) => write!(f, "vec4<{elem}>"),
            Self::Vec3(elem) => write!(f, "vec3<{elem}>"),
            Self::Vec2(elem) => write!(f, "vec2<{elem}>"),
            Self::Scalar(elem) => write!(f, "{elem}"),
        }
    }
}

fn format_number(num: f64) -> String {
    let formatted = format!("{:.34}", num);
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string() + "f"
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GlobalInputArray(number, _) => {
                write!(f, "input_{number}_global")
            }
            Self::LocalScalar {
                id: index,
                depth: scope_depth,
                ..
            } => write!(f, "s_{scope_depth}_{index}"),
            Self::Local {
                id: index,
                depth: scope_depth,
                ..
            } => write!(f, "l_{scope_depth}_{index}"),
            Self::LocalBinding { id: index, .. } => write!(f, "_{index}"),
            Self::Named { name, .. } => f.write_str(name),
            Self::Slice {
                id: index,
                item: _,
                depth: scope_depth,
            } => write!(f, "slice_{scope_depth}_{index}"),
            Self::GlobalOutputArray(number, _) => {
                write!(f, "output_{number}_global")
            }
            Self::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Self::ConstantScalar(number, _elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    IntKind::I32 => write!(f, "{}i", *val as i32),
                    IntKind::I64 => write!(f, "{}i", { *val }),
                },
                ConstantScalarValue::Float(val, kind) => match kind {
                    FloatKind::F16 => {
                        todo!("Unsupported")
                    }
                    FloatKind::BF16 => {
                        todo!("Unsupported")
                    }
                    FloatKind::F32 | FloatKind::F64 => f.write_str(&format_number(*val)),
                },
                ConstantScalarValue::UInt(val) => write!(f, "{}u", *val as u32),
                ConstantScalarValue::Bool(val) => write!(f, "{}", val),
            },
            Self::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }
            Self::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
            Self::LocalArray(number, _, scope_depth, _) => {
                write!(f, "a_{scope_depth}_{number}")
            }
            Self::Id => f.write_str("id"),
            Self::LocalInvocationIndex => f.write_str("local_idx"),
            Self::LocalInvocationIdX => f.write_str("local_invocation_id.x"),
            Self::LocalInvocationIdY => f.write_str("local_invocation_id.y"),
            Self::LocalInvocationIdZ => f.write_str("local_invocation_id.z"),
            Self::Rank => f.write_str("rank"),
            Self::WorkgroupId => f.write_str("workgroup_id_no_axis"),
            Self::WorkgroupIdX => f.write_str("workgroup_id.x"),
            Self::WorkgroupIdY => f.write_str("workgroup_id.y"),
            Self::WorkgroupIdZ => f.write_str("workgroup_id.z"),
            Self::GlobalInvocationIdX => f.write_str("global_id.x"),
            Self::GlobalInvocationIdY => f.write_str("global_id.y"),
            Self::GlobalInvocationIdZ => f.write_str("global_id.z"),
            Self::WorkgroupSizeX => f.write_str("WORKGROUP_SIZE_X"),
            Self::WorkgroupSizeY => f.write_str("WORKGROUP_SIZE_Y"),
            Self::WorkgroupSizeZ => f.write_str("WORKGROUP_SIZE_Z"),
            Self::NumWorkgroupsX => f.write_str("num_workgroups.x"),
            Self::NumWorkgroupsY => f.write_str("num_workgroups.y"),
            Self::NumWorkgroupsZ => f.write_str("num_workgroups.z"),
            Self::WorkgroupSize => f.write_str("workgroup_size_no_axis"),
            Self::NumWorkgroups => f.write_str("num_workgroups_no_axis"),
            Self::SubgroupSize => f.write_str("subgroup_size"),
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
            Self::LocalBinding { id, .. } => {
                format!("let _{id}")
            }
            var => format!("{}", var),
        }
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
