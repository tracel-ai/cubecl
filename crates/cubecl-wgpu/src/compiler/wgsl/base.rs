use cubecl_core::{
    ir::{ConstantValue, Id},
    prelude::Visibility,
};
use cubecl_ir::Intern;
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    GlobalBuffer(Id, Item),
    GlobalScalar(Id, Elem),
    Constant(ConstantValue, Item),
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
    Shared(Id, Item),
    ConstantArray(Id, Item, u32),
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
    SubgroupId,
    SubgroupInvocationId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum Elem {
    F16,
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum Item {
    Vector(Elem, usize),
    Scalar(Elem),
    Atomic(Intern<Item>),
    Pointer(Intern<Item>, PointerClass),
    Array(Intern<Item>, usize),
    DynamicArray(Intern<Item>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum PointerClass {
    Global(Visibility),
    Shared,
    Local,
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Variable {
    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::GlobalScalar(_, _) => true,
            Variable::Constant(_, _) => true,
            Variable::LocalScalar { .. } => true,
            Variable::Id => true,
            Variable::LocalInvocationIndex => true,
            Variable::LocalInvocationIdX => true,
            Variable::LocalInvocationIdY => true,
            Variable::LocalInvocationIdZ => true,
            Variable::GlobalBuffer(_, _) => false,
            Variable::Shared(_, _) => false,
            Variable::ConstantArray(_, _, _) => false,
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
            Variable::SubgroupId => true,
            Variable::SubgroupInvocationId => true,
        }
    }
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }

    pub fn is_ptr(&self) -> bool {
        self.item().is_ptr()
    }

    pub fn is_memory(&self) -> bool {
        matches!(self, Self::GlobalBuffer(..) | Self::Shared(..))
    }

    pub fn item(&self) -> Item {
        match self {
            Self::GlobalBuffer(_, e) => *e,
            Self::Shared(_, e) => *e,
            Self::ConstantArray(_, e, _) => *e,
            Self::LocalMut { item, .. } => *item,
            Self::LocalConst { item, .. } => *item,
            Self::Named { item, .. } => *item,
            Self::Constant(_, item) => *item,
            Self::GlobalScalar(_, e) => Item::Scalar(*e),
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
            Self::SubgroupId => Item::Scalar(Elem::U32),
            Self::SubgroupInvocationId => Item::Scalar(Elem::U32),
        }
    }
    pub fn elem(&self) -> Elem {
        *self.item().elem()
    }

    pub fn fmt_cast_to(&self, mut item: Item) -> String {
        while let Item::Pointer(inner, _) = item {
            item = *inner;
        }

        // Noop cast.
        if self.item() == item || self.is_ptr() {
            return format!("{self}");
        }

        let from = self.item();
        let from_elem = *from.elem();
        let to_elem = *item.elem();

        // Naga u64/i64 has weird limitations. We work around by first bitcasting to a 64-bit
        // type matching the target's signedness, then casting to the 32-bit target.
        let is_64bit = matches!(from_elem, Elem::I64 | Elem::U64);
        let is_32bit_target = matches!(to_elem, Elem::I32 | Elem::U32);
        if is_64bit && is_32bit_target {
            // Choose bitcast type based on target signedness (u32 -> u64, i32 -> i64)
            let bitcast_elem = if matches!(to_elem, Elem::U32) {
                Elem::U64
            } else {
                Elem::I64
            };

            if matches!(from, Item::Scalar(_)) {
                // Scalar cast (possibly splatted to vector)
                let scalar_cast = format!("{to_elem}(bitcast<{bitcast_elem}>({self}))");
                if matches!(item, Item::Scalar(_)) {
                    return scalar_cast;
                }
                return format!("{item}({scalar_cast})");
            }
            // Vector to vector cast
            let bitcast_item = from.with_elem(bitcast_elem);
            return format!("{item}(bitcast<{bitcast_item}>({self}))");
        }

        // WGSL doesn't support direct bool to f16 casts, can go through f32 first.
        if from_elem == Elem::Bool && to_elem == Elem::F16 {
            let f32_item = from.with_elem(Elem::F32);
            return format!("{item}({f32_item}({self}))");
        }

        // Default cases
        match (from, item) {
            // Scalar to scalar
            (Item::Scalar(_), Item::Scalar(_)) => format!("{item}({self})"),
            // Vec to scalar: pick first component
            (_, Item::Scalar(_)) => format!("{item}({self}.x)"),
            (Item::Scalar(_), _) if from_elem != to_elem => format!("{item}({to_elem}({self}))"),
            // Everything else (scalar to vec splat, vec to vec)
            _ => format!("{item}({self})"),
        }
    }
}

impl Item {
    pub fn intern(self) -> Intern<Self> {
        Intern::new(self)
    }

    pub fn elem(&self) -> &Elem {
        match self {
            Item::Scalar(e) => e,
            Item::Vector(elem, _) => elem,
            Item::Atomic(inner) => inner.elem(),
            Item::Pointer(inner, _) => inner.elem(),
            Item::Array(inner, _) => inner.elem(),
            Item::DynamicArray(inner) => inner.elem(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Item::Scalar(e) => e.size(),
            Item::Vector(elem, vector_size) => elem.size() * *vector_size,
            Item::Atomic(inner) => inner.size(),
            Item::Array(inner, length) => inner.size() * *length,
            Item::DynamicArray(inner) => inner.size(),
            Item::Pointer(..) => size_of::<u64>(),
        }
    }

    pub fn vectorization_factor(&self) -> usize {
        match self {
            Item::Scalar(_) => 1,
            Item::Vector(_, vector_size) => *vector_size,
            Item::Atomic(inner)
            | Item::Pointer(inner, _)
            | Item::Array(inner, _)
            | Item::DynamicArray(inner) => inner.vectorization_factor(),
        }
    }

    pub fn with_elem(self, elem: Elem) -> Self {
        match self {
            Item::Scalar(_) => Item::Scalar(elem),
            Item::Vector(_, vector_size) => Item::Vector(elem, vector_size),
            Item::Atomic(inner) => Item::Atomic(inner.with_elem(elem).intern()),
            Item::Pointer(inner, class) => Item::Pointer(inner.with_elem(elem).intern(), class),
            Item::Array(inner, size) => Item::Array(inner.with_elem(elem).intern(), size),
            Item::DynamicArray(inner) => Item::DynamicArray(inner.with_elem(elem).intern()),
        }
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, Item::Pointer(..))
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
            Self::I32 => core::mem::size_of::<i32>(),
            Self::I64 => core::mem::size_of::<i64>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::U64 => core::mem::size_of::<u64>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => f.write_str("f16"),
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
            Self::I32 => f.write_str("i32"),
            Self::I64 => f.write_str("i64"),
            Self::U32 => f.write_str("u32"),
            Self::U64 => f.write_str("u64"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Scalar(elem) => write!(f, "{elem}"),
            Item::Vector(elem, vector_size) => write!(f, "vec{vector_size}<{elem}>"),
            Item::Atomic(inner) => write!(f, "atomic<{inner}>"),
            Item::Pointer(inner, class) => match class {
                PointerClass::Global(Visibility::Uniform) => write!(f, "ptr<uniform, {inner}>"),
                PointerClass::Global(Visibility::Read) => write!(f, "ptr<storage, {inner}, read>"),
                PointerClass::Global(Visibility::ReadWrite) => {
                    write!(f, "ptr<storage, {inner}, read_write>")
                }
                PointerClass::Shared => write!(f, "ptr<workgroup, {inner}>"),
                PointerClass::Local => write!(f, "ptr<function, {inner}>"),
            },
            Item::Array(inner, size) => {
                write!(f, "array<{inner}, {size}>")
            }
            Item::DynamicArray(inner) => {
                write!(f, "array<{inner}>")
            }
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalBuffer(number, _) => {
                write!(f, "buffer_{number}_global")
            }
            Variable::LocalScalar { id: index, .. } => write!(f, "s_{index}"),
            Variable::LocalMut { id, .. } => write!(f, "l_mut_{id}"),
            Variable::LocalConst { id, .. } => write!(f, "l_{id}"),
            Variable::Named { name, .. } => f.write_str(name),
            Variable::GlobalScalar(number, elem) => {
                write!(f, "info.scalars_{elem}[{number}]")
            }
            Variable::Constant(val, item) => {
                match (val, item.elem()) {
                    // naga can't seem to parse literals > i64::MAX or i64::MIN atm.
                    // Work around this by emitting instructions to construct these literals.
                    (ConstantValue::UInt(v), Elem::U64) if *v > i64::MAX as u64 => {
                        let as_i64 = *v as i64;
                        if as_i64 == i64::MIN {
                            write!(f, "bitcast<u64>(i64(-9223372036854775807) - 1)")
                        } else {
                            write!(f, "bitcast<u64>(i64({as_i64}))")
                        }
                    }
                    (ConstantValue::Int(v), Elem::I64) if *v == i64::MIN => {
                        write!(f, "(i64(-9223372036854775807) - 1)")
                    }
                    (_, Elem::U64) | (_, Elem::I64) => write!(f, "{item}({val})"),
                    // For other cases we can just write the val with its type.
                    _ => write!(f, "{item}({val})"),
                }
            }
            Variable::Shared(number, _) => {
                write!(f, "shared_{number}")
            }
            Variable::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
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
            Variable::SubgroupId => f.write_str("subgroup_id"),
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
            Variable::GlobalScalar(_, _) => write!(f, "{var}"),
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
            var => format!("{var}"),
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
            Variable::GlobalScalar(_, _) => self.var.fmt_left(),
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
