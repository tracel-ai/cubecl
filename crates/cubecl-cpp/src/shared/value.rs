use cubecl_core::{
    e2m1, e2m1x2, e4m3, e5m2,
    ir::{ConstantValue, Id},
    ue8m0,
};
use cubecl_runtime::kernel::Visibility;
use std::fmt::{Display, Formatter};

use crate::shared::{FP4Kind, FP8Kind, PointerClass, binary::fmt_index};

use super::{COUNTER_TMP_VAR, Dialect, Elem, Item};

pub trait Component<D: Dialect>: Display + FmtLeft {
    fn item(&self) -> Item<D>;
    fn is_const(&self) -> bool;
    fn index(&self, index: usize) -> IndexedValue<D>;
    fn elem(&self) -> Elem<D> {
        *self.item().elem()
    }
}

pub trait FmtLeft: Display {
    fn fmt_left(&self) -> String;
}

#[derive(new, Debug)]
pub struct OptimizedArgs<const N: usize, D: Dialect> {
    pub args: [Value<D>; N],
    pub optimization_factor: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value<D: Dialect> {
    Constant(ConstantValue, Item<D>),
    Value {
        id: Id,
        item: Item<D>,
    },
    Tmp {
        id: Id,
        item: Item<D>,
        is_declared: bool,
        is_ptr: bool,
        is_const: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Builtin<D: Dialect> {
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
}

impl<D: Dialect> Builtin<D> {
    /// Format an item with a specific type, casting if necessary
    pub fn fmt_cast_to(&self, item: Item<D>) -> String {
        if self.item() == item {
            self.to_string()
        } else {
            format!("{item}({self})")
        }
    }

    pub fn item(&self) -> Item<D> {
        match self {
            Builtin::AbsolutePos(elem) => Item::Scalar(*elem),
            Builtin::AbsolutePosBaseName => Item::NativeVector(Elem::U32, 3),
            Builtin::AbsolutePosX => Item::Scalar(Elem::U32),
            Builtin::AbsolutePosY => Item::Scalar(Elem::U32),
            Builtin::AbsolutePosZ => Item::Scalar(Elem::U32),
            Builtin::CubeCount(elem) => Item::Scalar(*elem),
            Builtin::CubeCountBaseName => Item::NativeVector(Elem::U32, 3),
            Builtin::CubeCountX => Item::Scalar(Elem::U32),
            Builtin::CubeCountY => Item::Scalar(Elem::U32),
            Builtin::CubeCountZ => Item::Scalar(Elem::U32),
            Builtin::CubeDimBaseName => Item::NativeVector(Elem::U32, 3),
            Builtin::CubeDim => Item::Scalar(Elem::U32),
            Builtin::CubeDimX => Item::Scalar(Elem::U32),
            Builtin::CubeDimY => Item::Scalar(Elem::U32),
            Builtin::CubeDimZ => Item::Scalar(Elem::U32),
            Builtin::CubePos(elem) => Item::Scalar(*elem),
            Builtin::CubePosBaseName => Item::NativeVector(Elem::U32, 3),
            Builtin::CubePosX => Item::Scalar(Elem::U32),
            Builtin::CubePosY => Item::Scalar(Elem::U32),
            Builtin::CubePosZ => Item::Scalar(Elem::U32),
            Builtin::UnitPos => Item::Scalar(Elem::U32),
            Builtin::UnitPosBaseName => Item::NativeVector(Elem::U32, 3),
            Builtin::UnitPosX => Item::Scalar(Elem::U32),
            Builtin::UnitPosY => Item::Scalar(Elem::U32),
            Builtin::UnitPosZ => Item::Scalar(Elem::U32),
            Builtin::PlaneDim => Item::Scalar(Elem::U32),
            Builtin::PlaneDimChecked => Item::Scalar(Elem::U32),
            Builtin::PlanePos => Item::Scalar(Elem::U32),
            Builtin::UnitPosPlane => Item::Scalar(Elem::U32),
            Builtin::ClusterRank => Item::Scalar(Elem::U32),
            Builtin::ClusterIndexX => Item::Scalar(Elem::U32),
            Builtin::ClusterIndexY => Item::Scalar(Elem::U32),
            Builtin::ClusterIndexZ => Item::Scalar(Elem::U32),
        }
    }
}

impl<D: Dialect> Component<D> for Value<D> {
    fn index(&self, index: usize) -> IndexedValue<D> {
        self.index(index)
    }

    fn item(&self) -> Item<D> {
        match self {
            Value::Value { item, .. } => *item,
            Value::Constant(_, e) => *e,
            Value::Tmp { item, .. } => *item,
        }
    }

    fn is_const(&self) -> bool {
        if let Value::Tmp { is_const, .. } = self {
            return *is_const;
        }
        if let Item::Pointer(_, PointerClass::Global(Visibility::Read)) = self.item() {
            return true;
        }

        !self.item().is_ptr()
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

impl<D: Dialect> Display for Value<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Value { id, .. } => write!(f, "val_{id}"),
            Value::Constant(number, item) if item.vectorization() <= 1 => {
                let value = format_const(number, item);
                write!(f, "{item}({value})")
            }
            Value::Constant(number, item) => {
                let number = format_const(number, item);
                let values = (0..item.vectorization())
                    .map(|_| format!("{}({number})", item.elem()))
                    .collect::<Vec<_>>();
                write!(f, "{item} {{ {} }}", values.join(","))
            }
            Value::Tmp { id, .. } => write!(f, "_tmp_{id}"),
        }
    }
}

impl<D: Dialect> Display for Builtin<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Builtin::AbsolutePos(_) => D::compile_absolute_pos(f),
            Builtin::AbsolutePosBaseName => D::compile_absolute_pos_base_name(f),
            Builtin::AbsolutePosX => D::compile_absolute_pos_x(f),
            Builtin::AbsolutePosY => D::compile_absolute_pos_y(f),
            Builtin::AbsolutePosZ => D::compile_absolute_pos_z(f),
            Builtin::CubeCount(_) => D::compile_cube_count(f),
            Builtin::CubeCountBaseName => D::compile_cube_count_base_name(f),
            Builtin::CubeCountX => D::compile_cube_count_x(f),
            Builtin::CubeCountY => D::compile_cube_count_y(f),
            Builtin::CubeCountZ => D::compile_cube_count_z(f),
            Builtin::CubeDim => D::compile_cube_dim(f),
            Builtin::CubeDimBaseName => D::compile_cube_dim_base_name(f),
            Builtin::CubeDimX => D::compile_cube_dim_x(f),
            Builtin::CubeDimY => D::compile_cube_dim_y(f),
            Builtin::CubeDimZ => D::compile_cube_dim_z(f),
            Builtin::CubePos(_) => D::compile_cube_pos(f),
            Builtin::CubePosBaseName => D::compile_cube_pos_base_name(f),
            Builtin::CubePosX => D::compile_cube_pos_x(f),
            Builtin::CubePosY => D::compile_cube_pos_y(f),
            Builtin::CubePosZ => D::compile_cube_pos_z(f),
            Builtin::UnitPos => D::compile_unit_pos(f),
            Builtin::UnitPosBaseName => D::compile_unit_pos_base_name(f),
            Builtin::UnitPosX => D::compile_unit_pos_x(f),
            Builtin::UnitPosY => D::compile_unit_pos_y(f),
            Builtin::UnitPosZ => D::compile_unit_pos_z(f),
            Builtin::PlaneDim => D::compile_plane_dim(f),
            Builtin::PlaneDimChecked => D::compile_plane_dim_checked(f),
            Builtin::PlanePos => D::compile_plane_pos(f),
            Builtin::UnitPosPlane => D::compile_unit_pos_plane(f),
            Builtin::ClusterRank => D::compile_cluster_pos(f),
            Builtin::ClusterIndexX => D::compile_cluster_pos_x(f),
            Builtin::ClusterIndexY => D::compile_cluster_pos_y(f),
            Builtin::ClusterIndexZ => D::compile_cluster_pos_z(f),
        }
    }
}

impl<D: Dialect> Value<D> {
    pub fn is_optimized(&self) -> bool {
        self.item().is_optimized()
    }

    /// Create a temporary variable.
    ///
    /// Also see [`Self::tmp_declared`] for a version that needs custom declaration.
    pub fn tmp(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Value::Tmp {
            id: inc as Id,
            item,
            is_declared: false,
            is_ptr: false,
            is_const: false,
        }
    }

    pub fn to_const(&mut self) {
        if let Value::Tmp { is_const, .. } = self {
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
        let addr_space = D::address_space_for_value(self);
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

        Value::Tmp {
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
    /// Calling `val.fmt_left()` will assume the variable already exist.
    pub fn tmp_declared(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Value::Tmp {
            id: inc as Id,
            item,
            is_declared: true,
            is_ptr: false,
            is_const: false,
        }
    }

    pub fn optimized_args<const N: usize>(args: [Self; N]) -> OptimizedArgs<N, D> {
        let args_after = args.map(|a| a.optimized());

        let is_optimized = args_after.iter().all(|val| val.is_optimized());

        if is_optimized {
            let vectorization_before = args
                .iter()
                .map(|val| val.item().vectorization())
                .max()
                .unwrap();
            let vectorization_after = args_after
                .iter()
                .map(|val| val.item().vectorization())
                .max()
                .unwrap();

            OptimizedArgs::new(args_after, Some(vectorization_before / vectorization_after))
        } else {
            OptimizedArgs::new(args, None)
        }
    }

    pub fn optimized(&self) -> Self {
        match self {
            Value::Value { id, item } => Value::Value {
                id: *id,
                item: item.optimized(),
            },
            Value::Tmp {
                id,
                item,
                is_declared,
                is_ptr,
                is_const,
            } => Value::Tmp {
                id: *id,
                item: item.optimized(),
                is_declared: *is_declared,
                is_ptr: *is_ptr,
                is_const: *is_const,
            },
            _ => *self,
        }
    }

    pub fn index(&self, index: usize) -> IndexedValue<D> {
        IndexedValue {
            val: *self,
            index,
            optimized: self.is_optimized(),
        }
    }

    pub fn const_qualifier(&self) -> &str {
        if self.is_const() { " const" } else { "" }
    }

    pub fn id(&self) -> Option<Id> {
        match self {
            Value::Value { id, .. } => Some(*id),
            Value::Tmp { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Format variable for a pointer argument. Slices and buffers are already pointers, so we
    /// just leave them as is to avoid accidental double pointers
    pub fn fmt_ptr(&self) -> String {
        match self.item() {
            Item::Pointer(inner, _) if inner.is_array() => {
                format!("{self}->data")
            }
            Item::Array(..) => format!("{self}.data"),
            Item::DynamicArray(..) | Item::Pointer(..) => format!("{self}"),
            _ => format!("&{self}"),
        }
    }

    /// Format variable for a reference argument. Dereferences pointers while keeping locals as is.
    pub fn fmt_ref(&self) -> String {
        match self.item() {
            Item::DynamicArray(..) | Item::Pointer(..) => format!("*{self}"),
            _ => format!("{self}"),
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
    pub fn ensure_lvalue(&self, f: &mut Formatter<'_>) -> Result<Value<D>, core::fmt::Error> {
        if matches!(self, Value::Constant(..)) {
            let tmp = Value::tmp(self.item());
            writeln!(f, "{} = {self};", tmp.fmt_left())?;
            Ok(tmp)
        } else if matches!(self.item(), Item::Pointer(..)) {
            let tmp = Value::tmp(*self.item().value_ty());
            writeln!(f, "{}& {tmp} = *{self};", tmp.item())?;
            Ok(tmp)
        } else {
            Ok(*self)
        }
    }
}

impl<D: Dialect> FmtLeft for Value<D> {
    fn fmt_left(&self) -> String {
        match self {
            Self::Value { item, .. } => match item {
                // Pointer constness is determined by the type, not variable kind
                Item::Pointer(..) => {
                    format!("{item} {self}")
                }
                // Barrier is a memory object so can only exist behind a reference
                Item::Barrier(..) => {
                    format!("{item}& {self}")
                }
                // C++ has weird semantics so this needs to be mutable for use with `std::move`.
                // `std::move` preserves constness for the moved value, and the API requires
                // a non-const `BarrierToken&&`.
                Item::BarrierToken(..) => {
                    format!("{item} {self}")
                }
                _ => {
                    format!("const {item} {self}")
                }
            },
            Value::Tmp {
                item,
                is_declared,
                is_ptr,
                is_const,
                ..
            } => {
                if *is_declared {
                    return format!("{self}");
                }
                if *is_const && !*is_ptr {
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
pub struct IndexedValue<D: Dialect> {
    val: Value<D>,
    optimized: bool,
    index: usize,
}

impl<D: Dialect> Component<D> for IndexedValue<D> {
    fn item(&self) -> Item<D> {
        self.val.item()
    }

    fn index(&self, index: usize) -> IndexedValue<D> {
        self.val.index(index)
    }

    fn is_const(&self) -> bool {
        self.val.is_const()
    }
}

impl<D: Dialect> Display for IndexedValue<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.val;

        if let Value::Constant(value, item) = var {
            let value = format_const(value, item);
            return write!(f, "{}({value})", item.elem());
        }

        if var.item().unwrap_ptr().is_array_like() {
            return write!(f, "{}", fmt_index(var, &self.index, &var.item()));
        }

        let item = var.item();
        let addr_space = D::address_space_for_value(&self.val);
        let ty = match var {
            _ if item.is_ptr() => {
                format!("{item}")
            }
            Value::Value { item, .. } => format!("{addr_space}{item} const&"),
            _ => format!("{addr_space}{item}&"),
        };
        let accessor = match var.item().is_ptr() {
            true => "->",
            false => ".",
        };

        if self.val.item().vectorization() > 1 {
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

impl<D: Dialect> FmtLeft for IndexedValue<D> {
    fn fmt_left(&self) -> String {
        let var = &self.val;
        let ref_ = matches!(var, Value::Value { .. })
            .then_some("const&")
            .unwrap_or("&");

        let name = if self.val.item().vectorization() > 1 {
            if self.optimized {
                let item = self.val.item();
                let addr_space = D::address_space_for_value(&self.val);
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
            Value::Value { item, .. } => format!("const {item} {name}"),
            Value::Tmp { item, is_ptr, .. } => {
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
