use cubecl_core::ir::{self as gpu, BarrierLevel, ConstantScalarValue, Id};
use std::fmt::Display;

use super::{COUNTER_TMP_VAR, Dialect, Elem, Fragment, FragmentIdent, Item};

pub trait Component<D: Dialect>: Display + FmtLeft {
    fn item(&self) -> Item<D>;
    fn is_const(&self) -> bool;
    fn index(&self, index: usize) -> IndexedVariable<D>;
    fn elem(&self) -> Elem<D> {
        *self.item().elem()
    }
}

pub trait FmtLeft: Display {
    fn fmt_left(&self) -> String;
}

#[derive(new)]
pub struct OptimizedArgs<const N: usize, D: Dialect> {
    pub args: [Variable<D>; N],
    pub optimization_factor: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variable<D: Dialect> {
    AbsolutePos,
    AbsolutePosBaseName, // base name for XYZ
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
    UnitPos,
    UnitPosBaseName, // base name for XYZ
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePos,
    CubePosBaseName, // base name for XYZ
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimBaseName, // base name for XYZ
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeCount,
    CubeCountBaseName, // base name for XYZ
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    PlaneDim,
    PlaneDimChecked,
    PlanePos,
    UnitPosPlane,
    ClusterRank,
    ClusterIndexX,
    ClusterIndexY,
    ClusterIndexZ,
    GlobalInputArray(Id, Item<D>),
    GlobalOutputArray(Id, Item<D>),
    GlobalScalar {
        id: Id,
        elem: Elem<D>,
        in_struct: bool,
    },
    ConstantArray(Id, Item<D>, u32),
    ConstantScalar(ConstantScalarValue, Elem<D>),
    TensorMap(Id),
    LocalMut {
        id: Id,
        item: Item<D>,
    },
    LocalConst {
        id: Id,
        item: Item<D>,
    },
    Named {
        name: &'static str,
        item: Item<D>,
    },
    Slice {
        id: Id,
        item: Item<D>,
    },
    SharedMemory(Id, Item<D>, u32),
    LocalArray(Id, Item<D>, u32),
    WmmaFragment {
        id: Id,
        frag: Fragment<D>,
    },
    Pipeline {
        id: Id,
        item: Item<D>,
    },
    Barrier {
        id: Id,
        item: Item<D>,
        level: BarrierLevel,
    },
    Tmp {
        id: Id,
        item: Item<D>,
        is_declared: bool,
        is_ptr: bool,
    },
}

impl<D: Dialect> Component<D> for Variable<D> {
    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.index(index)
    }

    fn item(&self) -> Item<D> {
        match self {
            Variable::AbsolutePos => Item::scalar(Elem::U32, true),
            Variable::AbsolutePosBaseName => Item {
                elem: Elem::U32,
                vectorization: 3,
                native: true,
            },
            Variable::AbsolutePosX => Item::scalar(Elem::U32, true),
            Variable::AbsolutePosY => Item::scalar(Elem::U32, true),
            Variable::AbsolutePosZ => Item::scalar(Elem::U32, true),
            Variable::CubeCount => Item::scalar(Elem::U32, true),
            Variable::CubeCountBaseName => Item {
                elem: Elem::U32,
                vectorization: 3,
                native: true,
            },
            Variable::CubeCountX => Item::scalar(Elem::U32, true),
            Variable::CubeCountY => Item::scalar(Elem::U32, true),
            Variable::CubeCountZ => Item::scalar(Elem::U32, true),
            Variable::CubeDimBaseName => Item {
                elem: Elem::U32,
                vectorization: 3,
                native: true,
            },
            Variable::CubeDim => Item::scalar(Elem::U32, true),
            Variable::CubeDimX => Item::scalar(Elem::U32, true),
            Variable::CubeDimY => Item::scalar(Elem::U32, true),
            Variable::CubeDimZ => Item::scalar(Elem::U32, true),
            Variable::CubePos => Item::scalar(Elem::U32, true),
            Variable::CubePosBaseName => Item {
                elem: Elem::U32,
                vectorization: 3,
                native: true,
            },
            Variable::CubePosX => Item::scalar(Elem::U32, true),
            Variable::CubePosY => Item::scalar(Elem::U32, true),
            Variable::CubePosZ => Item::scalar(Elem::U32, true),
            Variable::UnitPos => Item::scalar(Elem::U32, true),
            Variable::UnitPosBaseName => Item {
                elem: Elem::U32,
                vectorization: 3,
                native: true,
            },
            Variable::UnitPosX => Item::scalar(Elem::U32, true),
            Variable::UnitPosY => Item::scalar(Elem::U32, true),
            Variable::UnitPosZ => Item::scalar(Elem::U32, true),
            Variable::PlaneDim => Item::scalar(Elem::U32, true),
            Variable::PlaneDimChecked => Item::scalar(Elem::U32, true),
            Variable::PlanePos => Item::scalar(Elem::U32, true),
            Variable::UnitPosPlane => Item::scalar(Elem::U32, true),
            Variable::ClusterRank => Item::scalar(Elem::U32, true),
            Variable::ClusterIndexX => Item::scalar(Elem::U32, true),
            Variable::ClusterIndexY => Item::scalar(Elem::U32, true),
            Variable::ClusterIndexZ => Item::scalar(Elem::U32, true),
            Variable::GlobalInputArray(_, e) => *e,
            Variable::GlobalOutputArray(_, e) => *e,
            Variable::LocalArray(_, e, _) => *e,
            Variable::SharedMemory(_, e, _) => *e,
            Variable::ConstantArray(_, e, _) => *e,
            Variable::LocalMut { item, .. } => *item,
            Variable::LocalConst { item, .. } => *item,
            Variable::Named { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::ConstantScalar(_, e) => Item::scalar(*e, false),
            Variable::GlobalScalar { elem, .. } => Item::scalar(*elem, false),
            Variable::WmmaFragment { frag, .. } => Item::scalar(frag.elem, false),
            Variable::Tmp { item, .. } => *item,
            Variable::Pipeline { id: _, item } => *item,
            Variable::Barrier { id: _, item, .. } => *item,
            Variable::TensorMap(_) => unreachable!(),
        }
    }

    fn is_const(&self) -> bool {
        matches!(
            self,
            Variable::LocalConst { .. } | Variable::GlobalInputArray(..)
        )
    }
}

impl<D: Dialect> Display for Variable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(id, _) => f.write_fmt(format_args!("buffer_{id}")),
            Variable::GlobalOutputArray(id, _) => write!(f, "buffer_{id}"),
            Variable::TensorMap(id) => write!(f, "tensor_map_{id}"),
            Variable::LocalMut { id, .. } => f.write_fmt(format_args!("l_mut_{id}")),
            Variable::LocalConst { id, .. } => f.write_fmt(format_args!("l_{id}")),
            Variable::Named { name, .. } => f.write_fmt(format_args!("{name}")),
            Variable::Slice { id, .. } => {
                write!(f, "slice_{id}")
            }
            Variable::GlobalScalar {
                id,
                elem,
                in_struct,
            } => match *in_struct {
                true => write!(f, "scalars_{elem}.x[{id}]"),
                false => write!(f, "scalars_{elem}[{id}]"),
            },
            Variable::ConstantScalar(number, elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    gpu::IntKind::I8 => write!(f, "{elem}({})", *val as i8),
                    gpu::IntKind::I16 => write!(f, "{elem}({})", *val as i16),
                    gpu::IntKind::I32 => write!(f, "{elem}({})", *val as i32),
                    gpu::IntKind::I64 => write!(f, "{elem}({})", *val),
                },
                ConstantScalarValue::Float(val, kind) => match kind {
                    gpu::FloatKind::E2M1
                    | gpu::FloatKind::E2M3
                    | gpu::FloatKind::E3M2
                    | gpu::FloatKind::E4M3
                    | gpu::FloatKind::E5M2
                    | gpu::FloatKind::UE8M0 => todo!("Minifloat constants not supported yet"),
                    gpu::FloatKind::F16 => {
                        write!(f, "{elem}({:?})", half::f16::from_f64(*val))
                    }
                    gpu::FloatKind::BF16 => {
                        write!(f, "{elem}({:?})", half::bf16::from_f64(*val))
                    }
                    gpu::FloatKind::Flex32 => write!(f, "{elem}({:?})", *val as f32),
                    gpu::FloatKind::TF32 => write!(f, "{elem}({:?})", *val as f32),
                    gpu::FloatKind::F32 => write!(f, "{elem}({:?})", *val as f32),
                    gpu::FloatKind::F64 => write!(f, "{elem}({:?})", *val),
                },
                ConstantScalarValue::UInt(val, kind) => match kind {
                    gpu::UIntKind::U8 => write!(f, "{elem}({})", *val as u8),
                    gpu::UIntKind::U16 => write!(f, "{elem}({})", *val as u16),
                    gpu::UIntKind::U32 => write!(f, "{elem}({})", *val as u32),
                    gpu::UIntKind::U64 => write!(f, "{elem}({})", *val),
                },
                ConstantScalarValue::Bool(val) => write!(f, "{val}"),
            },
            Variable::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }

            Variable::AbsolutePos => D::compile_absolute_pos(f),
            Variable::AbsolutePosBaseName => D::compile_absolute_pos_base_name(f),
            Variable::AbsolutePosX => D::compile_absolute_pos_x(f),
            Variable::AbsolutePosY => D::compile_absolute_pos_y(f),
            Variable::AbsolutePosZ => D::compile_absolute_pos_z(f),
            Variable::CubeCount => D::compile_cube_count(f),
            Variable::CubeCountBaseName => D::compile_cube_count_base_name(f),
            Variable::CubeCountX => D::compile_cube_count_x(f),
            Variable::CubeCountY => D::compile_cube_count_y(f),
            Variable::CubeCountZ => D::compile_cube_count_z(f),
            Variable::CubeDim => D::compile_cube_dim(f),
            Variable::CubeDimBaseName => D::compile_cube_dim_base_name(f),
            Variable::CubeDimX => D::compile_cube_dim_x(f),
            Variable::CubeDimY => D::compile_cube_dim_y(f),
            Variable::CubeDimZ => D::compile_cube_dim_z(f),
            Variable::CubePos => D::compile_cube_pos(f),
            Variable::CubePosBaseName => D::compile_cube_pos_base_name(f),
            Variable::CubePosX => D::compile_cube_pos_x(f),
            Variable::CubePosY => D::compile_cube_pos_y(f),
            Variable::CubePosZ => D::compile_cube_pos_z(f),
            Variable::UnitPos => D::compile_unit_pos(f),
            Variable::UnitPosBaseName => D::compile_unit_pos_base_name(f),
            Variable::UnitPosX => D::compile_unit_pos_x(f),
            Variable::UnitPosY => D::compile_unit_pos_y(f),
            Variable::UnitPosZ => D::compile_unit_pos_z(f),
            Variable::PlaneDim => D::compile_plane_dim(f),
            Variable::PlaneDimChecked => D::compile_plane_dim_checked(f),
            Variable::PlanePos => D::compile_plane_pos(f),
            Variable::UnitPosPlane => D::compile_unit_pos_plane(f),
            Variable::ClusterRank => D::compile_cluster_pos(f),
            Variable::ClusterIndexX => D::compile_cluster_pos_x(f),
            Variable::ClusterIndexY => D::compile_cluster_pos_y(f),
            Variable::ClusterIndexZ => D::compile_cluster_pos_z(f),

            Variable::ConstantArray(number, _, _) => f.write_fmt(format_args!("arrays_{number}")),
            Variable::LocalArray(id, _, _) => {
                write!(f, "l_arr_{id}")
            }
            Variable::WmmaFragment { id: index, frag } => {
                let name = match frag.ident {
                    FragmentIdent::A => "a",
                    FragmentIdent::B => "b",
                    FragmentIdent::Accumulator => "acc",
                    FragmentIdent::_Dialect(_) => "",
                };
                write!(f, "frag_{name}_{index}")
            }
            Variable::Tmp { id, .. } => write!(f, "_tmp_{id}"),
            Variable::Pipeline { id, .. } => write!(f, "pipeline_{id}"),
            Variable::Barrier { id, .. } => write!(f, "barrier_{id}"),
        }
    }
}

impl<D: Dialect> Variable<D> {
    pub fn is_optimized(&self) -> bool {
        self.item().is_optimized()
    }

    /// Create a temporary variable.
    ///
    /// Also see [Self::tmp_declared] for a version that needs custom declaration.
    pub fn tmp(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as Id,
            item,
            is_declared: false,
            is_ptr: false,
        }
    }

    /// Create a temporary pointer variable.
    ///
    /// Also see [Self::tmp_declared] for a version that needs custom declaration.
    pub fn tmp_ptr(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as Id,
            item,
            is_declared: false,
            is_ptr: true,
        }
    }

    /// Create a temporary variable with a custom declaration.
    ///
    /// # Notes
    ///
    /// Calling `var.fmt_left()` will assume the variable already exist.
    pub fn tmp_declared(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as Id,
            item,
            is_declared: true,
            is_ptr: false,
        }
    }

    pub fn optimized_args<const N: usize>(args: [Self; N]) -> OptimizedArgs<N, D> {
        let args_after = args.map(|a| a.optimized());

        let item_reference_after = args_after[0].item();

        let is_optimized = args_after
            .iter()
            .all(|var| var.elem() == item_reference_after.elem && var.is_optimized());

        if is_optimized {
            let vectorization_before = args
                .iter()
                .map(|var| var.item().vectorization)
                .max()
                .unwrap();
            let vectorization_after = args_after
                .iter()
                .map(|var| var.item().vectorization)
                .max()
                .unwrap();

            OptimizedArgs::new(args_after, Some(vectorization_before / vectorization_after))
        } else {
            OptimizedArgs::new(args, None)
        }
    }

    pub fn optimized(&self) -> Self {
        match self {
            Variable::GlobalInputArray(id, item) => {
                Variable::GlobalInputArray(*id, item.optimized())
            }
            Variable::GlobalOutputArray(id, item) => {
                Variable::GlobalOutputArray(*id, item.optimized())
            }
            Variable::LocalMut { id, item } => Variable::LocalMut {
                id: *id,
                item: item.optimized(),
            },
            Variable::LocalConst { id, item } => Variable::LocalConst {
                id: *id,
                item: item.optimized(),
            },
            Variable::Slice { id, item } => Variable::Slice {
                id: *id,
                item: item.optimized(),
            },
            Variable::Tmp {
                id,
                item,
                is_declared,
                is_ptr,
            } => Variable::Tmp {
                id: *id,
                item: item.optimized(),
                is_declared: *is_declared,
                is_ptr: *is_ptr,
            },
            Variable::SharedMemory(id, item, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Variable::SharedMemory(*id, item, size / scaling)
            }
            Variable::LocalArray(id, item, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Variable::LocalArray(*id, item.optimized(), size / scaling)
            }
            _ => *self,
        }
    }

    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::AbsolutePos => true,
            Variable::AbsolutePosBaseName => false,
            Variable::AbsolutePosX => true,
            Variable::AbsolutePosY => true,
            Variable::AbsolutePosZ => true,
            Variable::CubeCount => true,
            Variable::CubeCountBaseName => false,
            Variable::CubeCountX => true,
            Variable::CubeCountY => true,
            Variable::CubeCountZ => true,
            Variable::CubeDim => true,
            Variable::CubeDimBaseName => false,
            Variable::CubeDimX => true,
            Variable::CubeDimY => true,
            Variable::CubeDimZ => true,
            Variable::CubePos => true,
            Variable::CubePosBaseName => true,
            Variable::CubePosX => true,
            Variable::CubePosY => true,
            Variable::CubePosZ => true,
            Variable::UnitPos => true,
            Variable::UnitPosBaseName => true,
            Variable::UnitPosPlane => true,
            Variable::UnitPosX => true,
            Variable::UnitPosY => true,
            Variable::UnitPosZ => true,
            Variable::PlaneDim => true,
            Variable::PlaneDimChecked => true,
            Variable::PlanePos => true,
            Variable::ClusterRank => true,
            Variable::ClusterIndexX => true,
            Variable::ClusterIndexY => true,
            Variable::ClusterIndexZ => true,

            Variable::Barrier { .. } => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::ConstantScalar(_, _) => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::GlobalScalar { .. } => true,
            Variable::LocalArray(_, _, _) => false,
            Variable::LocalConst { .. } => false,
            Variable::LocalMut { .. } => false,
            Variable::Named { .. } => false,
            Variable::Pipeline { .. } => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::Slice { .. } => false,
            Variable::Tmp { .. } => false,
            Variable::WmmaFragment { .. } => false,
            Variable::TensorMap { .. } => false,
        }
    }

    pub fn index(&self, index: usize) -> IndexedVariable<D> {
        IndexedVariable {
            var: *self,
            index,
            optimized: self.is_optimized(),
        }
    }

    pub fn const_qualifier(&self) -> &str {
        if self.is_const() { " const" } else { "" }
    }

    pub fn id(&self) -> Option<Id> {
        match self {
            Variable::GlobalInputArray(id, ..) => Some(*id),
            Variable::GlobalOutputArray(id, ..) => Some(*id),
            Variable::GlobalScalar { id, .. } => Some(*id),
            Variable::ConstantArray(id, ..) => Some(*id),
            Variable::LocalMut { id, .. } => Some(*id),
            Variable::LocalConst { id, .. } => Some(*id),
            Variable::Slice { id, .. } => Some(*id),
            Variable::SharedMemory(id, ..) => Some(*id),
            Variable::LocalArray(id, ..) => Some(*id),
            Variable::WmmaFragment { id, .. } => Some(*id),
            Variable::Pipeline { id, .. } => Some(*id),
            Variable::Barrier { id, .. } => Some(*id),
            Variable::Tmp { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Format variable for a pointer argument. Slices and buffers are already pointers, so we
    /// just leave them as is to avoid accidental double pointers
    pub fn fmt_ptr(&self) -> String {
        match self {
            Variable::Slice { .. }
            | Variable::SharedMemory(_, _, _)
            | Variable::GlobalInputArray(_, _)
            | Variable::GlobalOutputArray(_, _) => format!("{self}"),
            _ => format!("&{self}"),
        }
    }
}

impl<D: Dialect> FmtLeft for Variable<D> {
    fn fmt_left(&self) -> String {
        match self {
            Self::LocalConst { item, .. } => format!("const {item} {self}"),
            Variable::Tmp {
                item,
                is_declared,
                is_ptr,
                ..
            } => {
                if *is_declared {
                    format!("{self}")
                } else if *is_ptr {
                    format!("{item} *{self}")
                } else {
                    format!("{item} {self}")
                }
            }
            var => format!("{var}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexedVariable<D: Dialect> {
    var: Variable<D>,
    optimized: bool,
    index: usize,
}

impl<D: Dialect> Component<D> for IndexedVariable<D> {
    fn item(&self) -> Item<D> {
        self.var.item()
    }

    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.var.index(index)
    }

    fn is_const(&self) -> bool {
        matches!(self.var, Variable::LocalConst { .. })
    }
}

impl<D: Dialect> Display for IndexedVariable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let ref_ = matches!(var, Variable::LocalConst { .. })
            .then_some("const&")
            .unwrap_or("&");

        if self.var.item().vectorization > 1 {
            if self.optimized {
                let item = self.var.item();
                let addr_space = D::address_space_for_variable(&self.var);
                write!(
                    f,
                    "(reinterpret_cast<{addr_space}{item} {ref_}>({var})).i_{}",
                    self.index
                )
            } else {
                write!(f, "{var}.i_{}", self.index)
            }
        } else if self.optimized {
            let item = self.var.item();
            let addr_space = D::address_space_for_variable(&self.var);
            write!(f, "reinterpret_cast<{addr_space}{item} {ref_}>({var})")
        } else {
            write!(f, "{var}")
        }
    }
}

impl<D: Dialect> FmtLeft for IndexedVariable<D> {
    fn fmt_left(&self) -> String {
        match self.var {
            Variable::LocalConst { item, .. } => format!("const {item} {self}"),
            Variable::Tmp { item, is_ptr, .. } => {
                if is_ptr {
                    format!("{item} *{self}")
                } else {
                    format!("{item} {self}")
                }
            }
            _ => format!("{self}"),
        }
    }
}

impl FmtLeft for &String {
    fn fmt_left(&self) -> String {
        self.to_string()
    }
}
