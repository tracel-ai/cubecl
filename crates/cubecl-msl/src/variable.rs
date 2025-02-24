use cubecl_core::ir as cube;
use crate::{BuiltInAttribute, Elem, Item};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    // Globals
    // -------
    GlobalInputArray(cube::Id, Item),
    GlobalOutputArray(cube::Id, Item),
    GlobalScalar(cube::Id, Elem, cube::Elem),
    // Cube built-ins
    // --------------
    // Variable with * denotes variables that are not provided by Metal specs.
    // Their values are computed in `format_cube_builtin_bindings_decl`.
    //
    // Units
    // thread position in grid
    ThreadIndexInGrid,                  // *
    ThreadPositionInGrid,
    ThreadPositionInGridX,
    ThreadPositionInGridY,
    ThreadPositionInGridZ,
    // thread count in threadgroup
    TotalThreadsInThreadgroup,          // *
    ThreadsPerThreadgoup,
    ThreadsPerThreadgoupX,
    ThreadsPerThreadgoupY,
    ThreadsPerThreadgoupZ,
    // thread position in threadgroup
    ThreadIndexInThreadgroup,
    ThreadPositionInThreadgroup,
    ThreadPositionInThreadgroupX,
    ThreadPositionInThreadgroupY,
    ThreadPositionInThreadgroupZ,
    // Cubes
    //
    // threadgroup count in grid
    TotalThreadgroupsInGrid,            // *
    ThreadgroupsPerGrid,
    ThreadgroupsPerGridX,
    ThreadgroupsPerGridY,
    ThreadgroupsPerGridZ,
    // threadgroup position in grid
    ThreadgroupIndexInGrid,             // *
    ThreadgroupPositionInGrid,
    ThreadgroupPositionInGridX,
    ThreadgroupPositionInGridY,
    ThreadgroupPositionInGridZ,
    // Planes
    //
    // simd-groups
    ThreadsPerSIMDgroup,
    ThreadIndexInSIMDgroup,



    ConstantArray(cube::Id, Item, u32),
    ConstantScalar(cube::ConstantScalarValue, Elem),
    LocalArray(cube::Id, Item, u32),
    LocalConst {id: cube::Id, item: Item},
    LocalMut {id: cube::Id, item: Item},
    // TODO: is is_array necessary
    Named {name: String, item: Item, is_array: bool},
    SharedMemory(cube::Id, Item, u32),
    Slice {id: cube::Id, item: Item},
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

fn format_number(num: f64) -> String {
    let formatted = format!("{:.34}", num);
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string() + "f"
}

impl Variable {
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
            Self::LocalMut { item, .. } => item.elem().is_atomic(),
            Self::Named { item, .. } => item.elem().is_atomic(),
            Self::Slice { item, .. } => item.elem().is_atomic(),
            Self::SharedMemory(_, item, _) => item.elem().is_atomic(),
            Self::LocalArray(_, item, _) => item.elem().is_atomic(),
            _ => false,
        }
    }

    pub fn fmt_left(&self) -> String {
        match self {
            Self::LocalConst { item, .. } => format!("const {item} {self}"),
            var => format!("{var}"),
        }
    }

    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Self::ThreadIndexInSIMDgroup => BuiltInAttribute::ThreadIndexInSIMDgroup,
            Self::ThreadIndexInThreadgroup => BuiltInAttribute::ThreadIndexInThreadgroup,
            Self::ThreadPositionInGrid => BuiltInAttribute::ThreadPositionInGrid,
            Self::ThreadPositionInThreadgroup => BuiltInAttribute::ThreadIndexInThreadgroup,
            Self::ThreadgroupPositionInGrid => BuiltInAttribute::ThreadgroupPositionInGrid,
            Self::ThreadgroupsPerGrid => BuiltInAttribute::ThreadgroupsPerGrid,
            Self::ThreadsPerSIMDgroup => BuiltInAttribute::ThreadsPerSIMDgroup,
            Self::ThreadsPerThreadgoup => BuiltInAttribute::ThreadsPerThreadgroup,
            _ => BuiltInAttribute::None,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            // Globals
            Variable::GlobalInputArray(_, e) => *e,
            Variable::GlobalOutputArray(_, e) => *e,
            Variable::GlobalScalar(_, e, _) => Item::Scalar(*e),
            // thread position in grid
            Self::ThreadIndexInGrid => Item::Scalar(Elem::U32),
            Self::ThreadPositionInGrid => Item::Vec3(Elem::U32),
            Self::ThreadPositionInGridX => Item::Scalar(Elem::U32),
            Self::ThreadPositionInGridY => Item::Scalar(Elem::U32),
            Self::ThreadPositionInGridZ => Item::Scalar(Elem::U32),
            // thread count in threadgroup
            Self::TotalThreadsInThreadgroup => Item::Scalar(Elem::U32),
            Self::ThreadsPerThreadgoup => Item::Vec3(Elem::U32),
            Self::ThreadsPerThreadgoupX => Item::Scalar(Elem::U32),
            Self::ThreadsPerThreadgoupY => Item::Scalar(Elem::U32),
            Self::ThreadsPerThreadgoupZ => Item::Scalar(Elem::U32),
            // thread position in threadgroup
            Self::ThreadIndexInThreadgroup => Item::Scalar(Elem::U32),
            Self::ThreadPositionInThreadgroup => Item::Vec3(Elem::U32),
            Self::ThreadPositionInThreadgroupX => Item::Scalar(Elem::U32),
            Self::ThreadPositionInThreadgroupY => Item::Scalar(Elem::U32),
            Self::ThreadPositionInThreadgroupZ => Item::Scalar(Elem::U32),
            // threadgroup count in grid
            Self::TotalThreadgroupsInGrid => Item::Scalar(Elem::U32),
            Self::ThreadgroupsPerGrid => Item::Vec3(Elem::U32),
            Self::ThreadgroupsPerGridX => Item::Scalar(Elem::U32),
            Self::ThreadgroupsPerGridY => Item::Scalar(Elem::U32),
            Self::ThreadgroupsPerGridZ => Item::Scalar(Elem::U32),
            // threadgroup position in grid
            Self::ThreadgroupIndexInGrid => Item::Scalar(Elem::U32),
            Self::ThreadgroupPositionInGrid => Item::Vec3(Elem::U32),
            Self::ThreadgroupPositionInGridX => Item::Scalar(Elem::U32),
            Self::ThreadgroupPositionInGridY => Item::Scalar(Elem::U32),
            Self::ThreadgroupPositionInGridZ => Item::Scalar(Elem::U32),
            // simd-groups
            Self::ThreadsPerSIMDgroup => Item::Scalar(Elem::U32),
            Self::ThreadIndexInSIMDgroup => Item::Scalar(Elem::U32),






            Variable::SharedMemory(_, e, _) => *e,
            Variable::ConstantArray(_, e, _) => *e,
            Variable::LocalArray(_, e, _) => *e,
            Variable::LocalMut { item, .. } => *item,
            Variable::LocalConst { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::Named { item, .. } => *item,
            Variable::ConstantScalar(_, e) => Item::Scalar(*e),
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

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
            Self::GlobalInputArray(number, _) => write!(f, "g_in_{number}"),
            Self::GlobalOutputArray(number, _) => write!(f, "g_out_{number}"),
            Self::GlobalScalar(number, _, elem) => write!(f, "scalars_{elem}[{number}]"),
            Self::LocalArray(number, _, _) => write!(f, "a_{number}"),
            Self::LocalConst { id, .. } => write!(f, "l_{id}"),
            Self::LocalMut { id, .. } => write!(f, "l_mut_{id}"),
            Self::Named { name, .. } => f.write_str(name),
            Self::SharedMemory(number, _, _) => write!(f, "shared_memory_{number}"),
            Self::Slice { id, .. } => write!(f, "slice_{id}"),
            // thread position in grid
            Self::ThreadIndexInGrid => f.write_str("thread_index_in_grid"),
            Self::ThreadPositionInGrid => f.write_str("thread_pos_in_grid"),
            Self::ThreadPositionInGridX => write!(f, "{}.x", Self::ThreadPositionInGrid),
            Self::ThreadPositionInGridY => write!(f, "{}.y", Self::ThreadPositionInGrid),
            Self::ThreadPositionInGridZ => write!(f, "{}.z", Self::ThreadPositionInGrid),
            // thread count in threadgroup
            Self::TotalThreadsInThreadgroup => f.write_str("total_thread_in_threadgroup"),
            Self::ThreadsPerThreadgoup => f.write_str("threads_per_threadgroup"),
            Self::ThreadsPerThreadgoupX => write!(f, "{}.x", Self::ThreadsPerThreadgoup),
            Self::ThreadsPerThreadgoupY => write!(f, "{}.y", Self::ThreadsPerThreadgoup),
            Self::ThreadsPerThreadgoupZ => write!(f, "{}.z", Self::ThreadsPerThreadgoup),
            // thread position in threadgroup
            Self::ThreadIndexInThreadgroup => f.write_str("thread_index_in_threadgroup"),
            Self::ThreadPositionInThreadgroup => f.write_str("thread_position_in_threadgroup"),
            Self::ThreadPositionInThreadgroupX => write!(f, "{}.x", Self::ThreadPositionInThreadgroup),
            Self::ThreadPositionInThreadgroupY => write!(f, "{}.y", Self::ThreadPositionInThreadgroup),
            Self::ThreadPositionInThreadgroupZ => write!(f, "{}.z", Self::ThreadPositionInThreadgroup),
            // threadgroup count in grid
            Self::TotalThreadgroupsInGrid => f.write_str("total_threadgroups_in_grid"),
            Self::ThreadgroupsPerGrid => f.write_str("threadgroups_per_grid"),
            Self::ThreadgroupsPerGridX => write!(f, "{}.x", Self::ThreadgroupsPerGrid),
            Self::ThreadgroupsPerGridY => write!(f, "{}.y", Self::ThreadgroupsPerGrid),
            Self::ThreadgroupsPerGridZ => write!(f, "{}.z", Self::ThreadgroupsPerGrid),
            // threadgroup position in grid
            Self::ThreadgroupIndexInGrid => f.write_str("threadgroup_index_in_grid"),
            Self::ThreadgroupPositionInGrid => f.write_str("threadgroup_position_in_grid"),
            Self::ThreadgroupPositionInGridX => write!(f, "{}.x", Self::ThreadgroupPositionInGrid),
            Self::ThreadgroupPositionInGridY => write!(f, "{}.y", Self::ThreadgroupPositionInGrid),
            Self::ThreadgroupPositionInGridZ => write!(f, "{}.z", Self::ThreadgroupPositionInGrid),
            // simd-groups
            Self::ThreadsPerSIMDgroup => f.write_str("threads_per_simdgroup"),
            Self::ThreadIndexInSIMDgroup => f.write_str("thread_index_in_simdgroup"),




            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Variable::ConstantScalar(number, _elem) => match number {
                cube::ConstantScalarValue::Int(val, kind) => match kind {
                    cube::IntKind::I32 => write!(f, "{}", *val as i32),
                    _ => unimplemented!("{:?} not supported in Metal language", kind),
                },
                cube::ConstantScalarValue::Float(val, kind) => match kind {
                    cube::FloatKind::F16 | cube::FloatKind::BF16 | cube::FloatKind::TF32 => {
                        todo!("Unsupported")
                    }
                    cube::FloatKind::F32 | cube::FloatKind::Flex32 | cube::FloatKind::F64 => {
                        f.write_str(&format_number(*val))
                    }
                },
                cube::ConstantScalarValue::UInt(val, _) => write!(f, "{}u", *val as u32),
                cube::ConstantScalarValue::Bool(val) => write!(f, "{}", val),
            },

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

impl IndexedVariable {
    pub fn fmt_left(&self) -> String {
        match self.var {
            Variable::LocalConst { item, .. } => format!("const {item} {self}"),
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
