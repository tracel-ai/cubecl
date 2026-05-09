use cubecl_core::{
    e2m1, e2m1x2, e4m3, e5m2,
    ir::{BarrierLevel, ConstantValue, Id},
    ue8m0,
};
use cubecl_runtime::kernel::Visibility;
use std::fmt::{Display, Formatter};

use crate::shared::{FP4Kind, FP8Kind, PointerClass};

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

#[derive(new, Debug)]
pub struct OptimizedArgs<const N: usize, D: Dialect> {
    pub args: [Variable<D>; N],
    pub optimization_factor: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variable<D: Dialect> {
    AbsolutePos(Elem<D>),
    AbsolutePosBaseName, // base name for XYZ
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
    UnitPos,
    UnitPosBaseName, // base name for XYZ
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePos(Elem<D>),
    CubePosBaseName, // base name for XYZ
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimBaseName, // base name for XYZ
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeCount(Elem<D>),
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
    GlobalBuffer(Id, Item<D>),
    GlobalScalar {
        id: Id,
        elem: Elem<D>,
    },
    ConstantArray(Id, Item<D>, usize),
    Constant(ConstantValue, Item<D>),
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
    SharedArray(Id, Item<D>, usize),
    Shared(Id, Item<D>),
    LocalArray(Id, Item<D>, usize),
    WmmaFragment {
        id: Id,
        frag: Fragment<D>,
    },
    Pipeline {
        id: Id,
    },
    Barrier {
        id: Id,
        level: BarrierLevel,
    },
    BarrierToken {
        id: Id,
        level: BarrierLevel,
    },
    Tmp {
        id: Id,
        item: Item<D>,
        is_declared: bool,
        is_ptr: bool,
        is_const: bool,
    },
}

impl<D: Dialect> Component<D> for Variable<D> {
    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.index(index)
    }

    fn item(&self) -> Item<D> {
        match self {
            Variable::AbsolutePos(elem) => Item::Scalar(*elem),
            Variable::AbsolutePosBaseName => Item::NativeVector(Elem::U32, 3),
            Variable::AbsolutePosX => Item::Scalar(Elem::U32),
            Variable::AbsolutePosY => Item::Scalar(Elem::U32),
            Variable::AbsolutePosZ => Item::Scalar(Elem::U32),
            Variable::CubeCount(elem) => Item::Scalar(*elem),
            Variable::CubeCountBaseName => Item::NativeVector(Elem::U32, 3),
            Variable::CubeCountX => Item::Scalar(Elem::U32),
            Variable::CubeCountY => Item::Scalar(Elem::U32),
            Variable::CubeCountZ => Item::Scalar(Elem::U32),
            Variable::CubeDimBaseName => Item::NativeVector(Elem::U32, 3),
            Variable::CubeDim => Item::Scalar(Elem::U32),
            Variable::CubeDimX => Item::Scalar(Elem::U32),
            Variable::CubeDimY => Item::Scalar(Elem::U32),
            Variable::CubeDimZ => Item::Scalar(Elem::U32),
            Variable::CubePos(elem) => Item::Scalar(*elem),
            Variable::CubePosBaseName => Item::NativeVector(Elem::U32, 3),
            Variable::CubePosX => Item::Scalar(Elem::U32),
            Variable::CubePosY => Item::Scalar(Elem::U32),
            Variable::CubePosZ => Item::Scalar(Elem::U32),
            Variable::UnitPos => Item::Scalar(Elem::U32),
            Variable::UnitPosBaseName => Item::NativeVector(Elem::U32, 3),
            Variable::UnitPosX => Item::Scalar(Elem::U32),
            Variable::UnitPosY => Item::Scalar(Elem::U32),
            Variable::UnitPosZ => Item::Scalar(Elem::U32),
            Variable::PlaneDim => Item::Scalar(Elem::U32),
            Variable::PlaneDimChecked => Item::Scalar(Elem::U32),
            Variable::PlanePos => Item::Scalar(Elem::U32),
            Variable::UnitPosPlane => Item::Scalar(Elem::U32),
            Variable::ClusterRank => Item::Scalar(Elem::U32),
            Variable::ClusterIndexX => Item::Scalar(Elem::U32),
            Variable::ClusterIndexY => Item::Scalar(Elem::U32),
            Variable::ClusterIndexZ => Item::Scalar(Elem::U32),
            Variable::GlobalBuffer(_, e) => *e,
            Variable::LocalArray(_, e, _) => *e,
            Variable::SharedArray(_, e, _) => *e,
            Variable::Shared(_, e) => *e,
            Variable::ConstantArray(_, e, _) => *e,
            Variable::LocalMut { item, .. } => *item,
            Variable::LocalConst { item, .. } => *item,
            Variable::Named { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::Constant(_, e) => *e,
            Variable::GlobalScalar { elem, .. } => Item::Scalar(*elem),
            Variable::WmmaFragment { frag, .. } => Item::Scalar(frag.elem),
            Variable::Tmp { item, .. } => *item,
            Variable::Pipeline { .. }
            | Variable::Barrier { .. }
            | Variable::BarrierToken { .. } => Item::Scalar(Elem::Bool),
            Variable::TensorMap(_) => unreachable!(),
        }
    }

    fn is_const(&self) -> bool {
        if let Variable::Tmp { is_const, .. } = self {
            return *is_const;
        }
        if let Item::Pointer(_, PointerClass::Global(Visibility::Read)) = self.item() {
            return true;
        }

        matches!(
            self,
            Variable::LocalConst { .. } | Variable::GlobalBuffer { .. }
        )
    }
}

pub(crate) fn format_const<D: Dialect>(number: &ConstantValue, item: &Item<D>) -> String {
    // minifloats are represented as raw bits, so use special handling
    let number = match item.elem() {
        Elem::FP4(FP4Kind::E2M1) => e2m1::from_f64(number.as_f64()).to_bits(),
        Elem::FP4x2(FP4Kind::E2M1) => {
            let v = number.as_f64() as f32;
            let value = [v, v];
            e2m1x2::from_f32_slice(&value).remove(0).to_bits()
        }
        Elem::FP6(_) | Elem::FP6x2(_) => {
            todo!("FP6 constants are not yet supported")
        }
        Elem::FP8(FP8Kind::E4M3) => e4m3::from_f64(number.as_f64()).to_bits(),
        Elem::FP8(FP8Kind::E5M2) => e5m2::from_f64(number.as_f64()).to_bits(),
        Elem::FP8(FP8Kind::UE8M0) => ue8m0::from_f64(number.as_f64()).to_bits(),
        _ => {
            return format!("{number}");
        }
    };
    format!("{number}")
}

impl<D: Dialect> Display for Variable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalBuffer(id, _) => write!(f, "buffer_{id}"),
            Variable::TensorMap(id) => write!(f, "tensor_map_{id}"),
            Variable::LocalMut { id, .. } => write!(f, "l_mut_{id}"),
            Variable::LocalConst { id, .. } => write!(f, "l_{id}"),
            Variable::Named { name, .. } => write!(f, "{name}"),
            Variable::Slice { id, .. } => {
                write!(f, "slice_{id}")
            }
            Variable::GlobalScalar { id, elem } => write!(f, "info.scalars_{elem}[{id}]"),
            Variable::Constant(number, item) if item.vectorization() <= 1 => {
                let value = format_const(number, item);
                write!(f, "{item}({value})")
            }
            Variable::Constant(number, item) => {
                let number = format_const(number, item);
                let values = (0..item.vectorization())
                    .map(|_| format!("{}({number})", item.elem()))
                    .collect::<Vec<_>>();
                write!(f, "{item} {{ {} }}", values.join(","))
            }
            Variable::SharedArray(number, _, _) | Variable::Shared(number, _) => {
                write!(f, "shared_memory_{number}")
            }

            Variable::AbsolutePos(_) => D::compile_absolute_pos(f),
            Variable::AbsolutePosBaseName => D::compile_absolute_pos_base_name(f),
            Variable::AbsolutePosX => D::compile_absolute_pos_x(f),
            Variable::AbsolutePosY => D::compile_absolute_pos_y(f),
            Variable::AbsolutePosZ => D::compile_absolute_pos_z(f),
            Variable::CubeCount(_) => D::compile_cube_count(f),
            Variable::CubeCountBaseName => D::compile_cube_count_base_name(f),
            Variable::CubeCountX => D::compile_cube_count_x(f),
            Variable::CubeCountY => D::compile_cube_count_y(f),
            Variable::CubeCountZ => D::compile_cube_count_z(f),
            Variable::CubeDim => D::compile_cube_dim(f),
            Variable::CubeDimBaseName => D::compile_cube_dim_base_name(f),
            Variable::CubeDimX => D::compile_cube_dim_x(f),
            Variable::CubeDimY => D::compile_cube_dim_y(f),
            Variable::CubeDimZ => D::compile_cube_dim_z(f),
            Variable::CubePos(_) => D::compile_cube_pos(f),
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

            Variable::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
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
            Variable::BarrierToken { id, .. } => write!(f, "barrier_{id}_token"),
        }
    }
}

impl<D: Dialect> Variable<D> {
    pub fn is_optimized(&self) -> bool {
        self.item().is_optimized()
    }

    /// Create a temporary variable.
    ///
    /// Also see [`Self::tmp_declared`] for a version that needs custom declaration.
    pub fn tmp(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as Id,
            item,
            is_declared: false,
            is_ptr: false,
            is_const: false,
        }
    }

    pub fn to_const(&mut self) {
        if let Variable::Tmp { is_const, .. } = self {
            *is_const = true;
        }
    }

    /// Create a temporary variable with a `reinterpret_cast`.
    pub fn reinterpret_ptr(&self, f: &mut Formatter<'_>, item: Item<D>) -> Self {
        let mut out = Self::tmp_ptr(item);

        if self.is_const() {
            out.to_const();
        }

        let elem = out.elem();
        let qualifier = out.const_qualifier();
        let addr_space = D::address_space_for_variable(self);
        let out_fmt = out.fmt_left();

        writeln!(
            f,
            "{out_fmt} = reinterpret_cast<{addr_space}{elem}{qualifier}*>({self});"
        )
        .unwrap();

        out
    }

    /// Create a temporary pointer variable.
    ///
    /// Also see [`Self::tmp_declared`] for a version that needs custom declaration.
    pub fn tmp_ptr(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as Id,
            item,
            is_declared: false,
            is_ptr: true,
            is_const: false,
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
            is_const: false,
        }
    }

    pub fn optimized_args<const N: usize>(args: [Self; N]) -> OptimizedArgs<N, D> {
        let args_after = args.map(|a| a.optimized());

        let is_optimized = args_after.iter().all(|var| var.is_optimized());

        if is_optimized {
            let vectorization_before = args
                .iter()
                .map(|var| var.item().vectorization())
                .max()
                .unwrap();
            let vectorization_after = args_after
                .iter()
                .map(|var| var.item().vectorization())
                .max()
                .unwrap();

            OptimizedArgs::new(args_after, Some(vectorization_before / vectorization_after))
        } else {
            OptimizedArgs::new(args, None)
        }
    }

    pub fn optimized(&self) -> Self {
        match self {
            Variable::GlobalBuffer(id, item) => Variable::GlobalBuffer(*id, item.optimized()),
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
                is_const,
            } => Variable::Tmp {
                id: *id,
                item: item.optimized(),
                is_declared: *is_declared,
                is_ptr: *is_ptr,
                is_const: *is_const,
            },
            Variable::SharedArray(id, item, size) => {
                let before = item.vectorization();
                let item = item.optimized();
                let after = item.vectorization();
                let scaling = before / after;

                Variable::SharedArray(*id, item, size / scaling)
            }
            Variable::LocalArray(id, item, size) => {
                let before = item.vectorization();
                let item = item.optimized();
                let after = item.vectorization();
                let scaling = before / after;

                Variable::LocalArray(*id, item.optimized(), size / scaling)
            }
            _ => *self,
        }
    }

    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::AbsolutePos(_) => true,
            Variable::AbsolutePosBaseName => false,
            Variable::AbsolutePosX => true,
            Variable::AbsolutePosY => true,
            Variable::AbsolutePosZ => true,
            Variable::CubeCount(_) => true,
            Variable::CubeCountBaseName => false,
            Variable::CubeCountX => true,
            Variable::CubeCountY => true,
            Variable::CubeCountZ => true,
            Variable::CubeDim => true,
            Variable::CubeDimBaseName => false,
            Variable::CubeDimX => true,
            Variable::CubeDimY => true,
            Variable::CubeDimZ => true,
            Variable::CubePos(_) => true,
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
            Variable::BarrierToken { .. } => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::Constant(_, _) => true,
            Variable::GlobalBuffer(_, _) => false,
            Variable::GlobalScalar { .. } => true,
            Variable::LocalArray(_, _, _) => false,
            Variable::LocalConst { .. } => false,
            Variable::LocalMut { .. } => false,
            Variable::Named { .. } => false,
            Variable::Pipeline { .. } => false,
            Variable::SharedArray(_, _, _) => false,
            Variable::Shared(_, _) => false,
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
            Variable::GlobalBuffer(id, ..) => Some(*id),
            Variable::GlobalScalar { id, .. } => Some(*id),
            Variable::ConstantArray(id, ..) => Some(*id),
            Variable::LocalMut { id, .. } => Some(*id),
            Variable::LocalConst { id, .. } => Some(*id),
            Variable::Slice { id, .. } => Some(*id),
            Variable::Shared(id, ..) => Some(*id),
            Variable::SharedArray(id, ..) => Some(*id),
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
            | Variable::SharedArray(_, _, _)
            | Variable::GlobalBuffer(_, _) => format!("{self}"),
            other => match other.item() {
                Item::Array(..) => format!("{other}.data"),
                Item::DynamicArray(..) | Item::Pointer(..) => format!("{other}"),
                _ => format!("&{other}"),
            },
        }
    }

    /// Format an item with a specific type, casting if necessary
    pub fn fmt_cast_to(&self, item: Item<D>) -> String {
        if self.item() == item {
            self.to_string()
        } else {
            format!("{item}({self})")
        }
    }

    /// Ensure a variable is a named lvalue, reassigning to a temporary if necessary.
    /// This is required for reinterpreting constants.
    pub fn ensure_lvalue(&self, f: &mut Formatter<'_>) -> Result<Variable<D>, core::fmt::Error> {
        if matches!(self, Variable::Constant(..)) {
            let mut tmp = Variable::tmp(self.item());
            tmp.to_const();
            writeln!(f, "{} = {self};", tmp.fmt_left())?;
            Ok(tmp)
        } else {
            Ok(*self)
        }
    }
}

impl<D: Dialect> FmtLeft for Variable<D> {
    fn fmt_left(&self) -> String {
        match self {
            Self::LocalConst { item, .. } => match item {
                // Pointer constness is determined by the type, not variable kind
                Item::Pointer(..) => {
                    format!("{item} {self}")
                }
                _ => {
                    format!("const {item} {self}")
                }
            },
            Variable::Tmp {
                item,
                is_declared,
                is_ptr,
                is_const,
                ..
            } => {
                if *is_declared {
                    return format!("{self}");
                }
                if *is_ptr {
                    if *is_const {
                        return format!("const {item} *{self}");
                    }
                    return format!("{item} *{self}");
                }
                if *is_const {
                    format!("const {item} {self}")
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
        self.var.is_const()
    }
}

impl<D: Dialect> Display for IndexedVariable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;

        if let Variable::Constant(value, item) = var {
            let value = format_const(value, item);
            return write!(f, "{}({value})", item.elem());
        }

        let item = var.item();
        let addr_space = D::address_space_for_variable(&self.var);
        let ty = match var {
            _ if item.is_ptr() => {
                format!("{item}")
            }
            Variable::LocalConst { item, .. } => format!("{addr_space}{item} const&"),
            _ => format!("{addr_space}{item}&"),
        };
        let accessor = match var.item().is_ptr() {
            true => "->",
            false => ".",
        };

        if self.var.item().vectorization() > 1 {
            if self.optimized {
                write!(
                    f,
                    "(reinterpret_cast<{ty}>({var})){accessor}i_{}",
                    self.index
                )
            } else {
                write!(f, "{var}{accessor}i_{}", self.index)
            }
        } else if self.optimized {
            write!(f, "reinterpret_cast<{ty}>({var})")
        } else {
            write!(f, "{var}")
        }
    }
}

impl<D: Dialect> FmtLeft for IndexedVariable<D> {
    fn fmt_left(&self) -> String {
        let var = &self.var;
        let ref_ = matches!(var, Variable::LocalConst { .. })
            .then_some("const&")
            .unwrap_or("&");

        let name = if self.var.item().vectorization() > 1 {
            if self.optimized {
                let item = self.var.item();
                let addr_space = D::address_space_for_variable(&self.var);
                format!(
                    "(reinterpret_cast<{addr_space}{item} {ref_}>({var})).i_{}",
                    self.index
                )
            } else {
                format!("{var}.i_{}", self.index)
            }
        } else {
            format!("{var}")
        };
        match var {
            Variable::LocalConst { item, .. } => format!("const {item} {name}"),
            Variable::Tmp { item, is_ptr, .. } => {
                if *is_ptr {
                    format!("{item} *{name}")
                } else {
                    format!("{item} {name}")
                }
            }
            _ => name,
        }
    }
}

impl FmtLeft for &String {
    fn fmt_left(&self) -> String {
        self.to_string()
    }
}
