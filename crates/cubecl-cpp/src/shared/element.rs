use cubecl_core::ir::{self as gpu, ConstantScalarValue};
use half::{bf16, f16};
use std::fmt::Display;

use super::{Dialect, Fragment, COUNTER_TMP_VAR};

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum Elem<D: Dialect> {
    F32,
    F16,
    F162,
    BF16,
    BF162,
    I32,
    U32,
    Bool,
    Atomic(AtomicKind),
    _Dialect(std::marker::PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum AtomicKind {
    I32,
    U32,
}

impl Display for AtomicKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomicKind::I32 => f.write_str("int"),
            AtomicKind::U32 => f.write_str("uint"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Item<D: Dialect> {
    pub(crate) elem: Elem<D>,
    pub(crate) vectorization: usize,
}

impl<D: Dialect> Display for Elem<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::F16 => f.write_str("__half"),
            Elem::F162 => f.write_str("__half2"),
            Elem::F32 => f.write_str("float"),
            Elem::BF16 => D::bfloat16_type_name(f),
            Elem::BF162 => D::bfloat162_type_name(f),
            Elem::I32 => f.write_str("int"),
            Elem::U32 => f.write_str("uint"),
            Elem::Bool => f.write_str("bool"),
            Elem::Atomic(inner) => inner.fmt(f),
            Elem::_Dialect(_) => Ok(()),
        }
    }
}

impl<D: Dialect> Display for Item<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if 1 == self.vectorization {
            return write!(f, "{}", self.elem);
        }

        write!(f, "{}_{}", self.elem, self.vectorization)
    }
}

pub trait Component<D: Dialect>: Display + FmtLeft {
    fn item(&self) -> Item<D>;
    fn is_const(&self) -> bool;
    fn index(&self, index: usize) -> IndexedVariable<D>;
    fn elem(&self) -> Elem<D> {
        *self.item().elem()
    }
}

impl<D: Dialect> Component<D> for IndexedVariable<D> {
    fn item(&self) -> Item<D> {
        self.var.item()
    }

    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.var.index(index)
    }

    fn is_const(&self) -> bool {
        matches!(self.var, Variable::ConstLocal { .. })
    }
}

impl<D: Dialect> Component<D> for Variable<D> {
    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.index(index)
    }

    fn item(&self) -> Item<D> {
        match self {
            Self::GlobalInputArray(_, e)
            | Self::GlobalOutputArray(_, e)
            | Self::SharedMemory(_, e, _)
            | Self::ConstantArray(_, e, _) => *e,
            Self::Local { item, .. } | Self::ConstLocal { item, .. } | Self::Slice { item, .. } => {
                *item
            }
            Self::ConstantScalar(_, e) | Self::GlobalScalar(_, e, _) => Item::scalar(*e),
            Self::IdxGlobal
            | Self::ThreadIdxGlobal
            | Self::ThreadIdxX
            | Self::ThreadIdxY
            | Self::ThreadIdxZ
            | Self::Rank
            | Self::BlockIdxX
            | Self::BlockIdxY
            | Self::BlockIdxZ
            | Self::AbsoluteIdxX
            | Self::AbsoluteIdxY
            | Self::AbsoluteIdxZ
            | Self::BlockDimX
            | Self::BlockDimY
            | Self::BlockDimZ
            | Self::GridDimX
            | Self::GridDimY
            | Self::BlockIdxGlobal
            | Self::BlockDimGlobal
            | Self::GridDimGlobal
            | Self::WarpSize
            | Self::GridDimZ => Item::scalar(Elem::U32),
            Self::LocalArray(_, e, _, _) => *e,

            Self::WmmaFragment { frag, .. } => Item::scalar(frag.elem),

            Self::Tmp { item, .. } => *item,
        }
    }

    fn is_const(&self) -> bool {
        matches!(self, Variable::ConstLocal { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variable<D: Dialect> {
    WarpSize,
    GlobalInputArray(u16, Item<D>),
    GlobalOutputArray(u16, Item<D>),
    GlobalScalar(u16, Elem<D>, gpu::Elem),
    ConstantArray(u16, Item<D>, u32),
    ConstantScalar(ConstantScalarValue, Elem<D>),
    Local {
        id: u16,
        item: Item<D>,
        depth: u8,
    },
    ConstLocal {
        id: u16,
        item: Item<D>,
        depth: u8,
    },
    Slice {
        id: u16,
        item: Item<D>,
        depth: u8,
    },
    SharedMemory(u16, Item<D>, u32),
    LocalArray(u16, Item<D>, u8, u32),
    IdxGlobal,
    ThreadIdxGlobal,
    ThreadIdxX,
    ThreadIdxY,
    ThreadIdxZ,
    Rank,
    BlockIdxGlobal,
    BlockIdxX,
    BlockIdxY,
    BlockIdxZ,
    AbsoluteIdxX,
    AbsoluteIdxY,
    AbsoluteIdxZ,
    BlockDimGlobal,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimGlobal,
    GridDimX,
    GridDimY,
    GridDimZ,
    WmmaFragment {
        id: u16,
        frag: Fragment<D>,
        depth: u8,
    },
    Tmp {
        id: u16,
        item: Item<D>,
    },
}

impl<D: Dialect> Display for Variable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GlobalInputArray(number, _) => f.write_fmt(format_args!("input_{number}")),
            Self::Local { id, depth, .. } => f.write_fmt(format_args!("l_{depth}_{id}")),
            Self::ConstLocal { id, depth, .. } => f.write_fmt(format_args!("ssa_{depth}_{id}")),
            Self::Slice { id, depth, .. } => {
                write!(f, "slice_{depth}_{id}")
            }
            Self::GlobalOutputArray(number, _) => write!(f, "output_{number}"),
            Self::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Self::ConstantScalar(number, elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    gpu::IntKind::I32 => write!(f, "{elem}({})", *val as i32),
                    gpu::IntKind::I64 => write!(f, "{elem}({})", *val),
                },
                ConstantScalarValue::Float(val, kind) => match kind {
                    gpu::FloatKind::F16 => {
                        write!(f, "{elem}({:?})", half::f16::from_f64(*val))
                    }
                    gpu::FloatKind::BF16 => {
                        write!(f, "{elem}({:?})", half::bf16::from_f64(*val))
                    }
                    gpu::FloatKind::F32 => write!(f, "{elem}({:?})", *val as f32),
                    gpu::FloatKind::F64 => write!(f, "{elem}({:?})", *val),
                },
                ConstantScalarValue::UInt(val) => {
                    write!(f, "{elem}({})", *val as u32)
                }
                ConstantScalarValue::Bool(val) => write!(f, "{}", val),
            },
            Self::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }
            Self::ConstantArray(number, _, _) => f.write_fmt(format_args!("arrays_{number}")),
            Self::ThreadIdxGlobal => f.write_str("threadIdxGlobal"),
            Self::ThreadIdxX => f.write_str("threadIdx.x"),
            Self::ThreadIdxY => f.write_str("threadIdx.y"),
            Self::ThreadIdxZ => f.write_str("threadIdx.z"),
            Self::Rank => f.write_str("rank"),
            Self::BlockIdxGlobal => f.write_str("blockIdxGlobal"),
            Self::BlockIdxX => f.write_str("blockIdx.x"),
            Self::BlockIdxY => f.write_str("blockIdx.y"),
            Self::BlockIdxZ => f.write_str("blockIdx.z"),
            Self::BlockDimGlobal => f.write_str("blockDimGlobal"),
            Self::BlockDimX => f.write_str("blockDim.x"),
            Self::BlockDimY => f.write_str("blockDim.y"),
            Self::BlockDimZ => f.write_str("blockDim.z"),
            Self::IdxGlobal => f.write_str("idxGlobal"),
            Self::GridDimX => f.write_str("gridDim.x"),
            Self::GridDimY => f.write_str("gridDim.y"),
            Self::GridDimZ => f.write_str("gridDim.z"),
            Self::AbsoluteIdxX => f.write_str("absoluteIdx.x"),
            Self::AbsoluteIdxY => f.write_str("absoluteIdx.y"),
            Self::AbsoluteIdxZ => f.write_str("absoluteIdx.z"),
            Self::LocalArray(id, _item, depth, _size) => {
                write!(f, "l_arr_{}_{}", id, depth)
            }
            Self::WarpSize => f.write_str("warpSize"),
            Self::WmmaFragment {
                id: index, depth, ..
            } => write!(f, "frag_{index}_{depth}"),
            Self::GridDimGlobal => f.write_str("gridDimGlobal"),
            Self::Tmp { id, .. } => write!(f, "_tmp_{id}"),
        }
    }
}

#[derive(new)]
pub struct OptimizedArgs<const N: usize, D: Dialect> {
    pub args: [Variable<D>; N],
    pub optimization_factor: Option<usize>,
}

impl<D: Dialect> Variable<D> {
    pub fn is_optimized(&self) -> bool {
        self.item().is_optimized()
    }

    pub fn tmp(item: Item<D>) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as u16,
            item,
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
            Self::GlobalInputArray(id, item) => Self::GlobalInputArray(*id, item.optimized()),
            Self::GlobalOutputArray(id, item) => Self::GlobalOutputArray(*id, item.optimized()),
            Self::Local { id, item, depth } => Self::Local {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Self::ConstLocal { id, item, depth } => Self::ConstLocal {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Self::Slice { id, item, depth } => Self::Slice {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Self::SharedMemory(id, item, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Self::SharedMemory(*id, item, size / scaling)
            }
            Self::LocalArray(id, item, vec, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Self::LocalArray(*id, item.optimized(), *vec, size / scaling)
            }
            _ => *self,
        }
    }

    pub fn is_always_scalar(&self) -> bool {
        match self {
            Self::ConstantArray(_, _, _)
            | Self::ConstLocal { .. }
            | Self::GlobalInputArray(_, _)
            | Self::GlobalOutputArray(_, _)
            | Self::Local { .. }
            | Self::LocalArray(_, _, _, _)
            | Self::SharedMemory(_, _, _)
            | Self::Slice { .. }
            | Self::Tmp { .. }
            | Self::WmmaFragment { .. } => false,
            Self::AbsoluteIdxX
            | Self::AbsoluteIdxY
            | Self::AbsoluteIdxZ
            | Self::BlockDimGlobal
            | Self::BlockDimX
            | Self::BlockDimY
            | Self::BlockDimZ
            | Self::BlockIdxGlobal
            | Self::BlockIdxX
            | Self::BlockIdxY
            | Self::BlockIdxZ
            | Self::ConstantScalar(_, _)
            | Self::GlobalScalar(_, _, _)
            | Self::GridDimGlobal
            | Self::GridDimX
            | Self::GridDimY
            | Self::GridDimZ
            | Self::IdxGlobal
            | Self::Rank
            | Self::ThreadIdxGlobal
            | Self::ThreadIdxX
            | Self::ThreadIdxY
            | Self::ThreadIdxZ
            | Self::WarpSize => true,
        }
    }

    pub fn index(&self, index: usize) -> IndexedVariable<D> {
        IndexedVariable {
            var: *self,
            index,
            optimized: self.is_optimized(),
        }
    }
}

pub trait FmtLeft: Display {
    fn fmt_left(&self) -> String;
}

impl<D: Dialect> FmtLeft for Variable<D> {
    fn fmt_left(&self) -> String {
        match self {
            Self::ConstLocal { item, .. } => format!("const {item} {self}"),
            Self::Tmp { item, .. } => format!("{item} {self}"),
            var => format!("{var}"),
        }
    }
}

impl<D: Dialect> FmtLeft for IndexedVariable<D> {
    fn fmt_left(&self) -> String {
        match self.var {
            Variable::ConstLocal { item, .. } => format!("const {item} {self}"),
            Variable::Tmp { item, .. } => format!("{item} {self}"),
            _ => format!("{self}"),
        }
    }
}

impl FmtLeft for &String {
    fn fmt_left(&self) -> String {
        self.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct IndexedVariable<D: Dialect> {
    var: Variable<D>,
    optimized: bool,
    index: usize,
}

impl<D: Dialect> Display for IndexedVariable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let ref_ = matches!(var, Variable::ConstLocal { .. })
            .then_some("const&")
            .unwrap_or("&");

        if self.var.item().vectorization > 1 {
            if self.optimized {
                let item = self.var.item();
                write!(
                    f,
                    "(reinterpret_cast<{item} {ref_}>({var})).i_{}",
                    self.index
                )
            } else {
                write!(f, "{var}.i_{}", self.index)
            }
        } else if self.optimized {
            let item = self.var.item();
            write!(f, "reinterpret_cast<{item} {ref_}>({var})")
        } else {
            write!(f, "{var}")
        }
    }
}
impl<D: Dialect> Item<D> {
    pub fn elem(&self) -> &Elem<D> {
        &self.elem
    }

    pub fn de_optimized(&self) -> Self {
        match self.elem {
            Elem::F162 => Item::new(Elem::F16, self.vectorization * 2),
            Elem::BF162 => Item::new(Elem::BF16, self.vectorization * 2),
            _ => *self,
        }
    }

    pub fn new(elem: Elem<D>, vectorization: usize) -> Self {
        Self {
            elem,
            vectorization,
        }
    }
    pub fn scalar(elem: Elem<D>) -> Self {
        Self {
            elem,
            vectorization: 1,
        }
    }

    pub fn is_optimized(&self) -> bool {
        matches!(self.elem, Elem::F162 | Elem::BF162)
    }

    pub fn optimized(&self) -> Item<D> {
        if self.vectorization == 1 {
            return *self;
        }

        if self.vectorization % 2 != 0 {
            return *self;
        }

        match self.elem {
            Elem::F16 => Item {
                elem: Elem::F162,
                vectorization: self.vectorization / 2,
            },
            Elem::BF16 => Item {
                elem: Elem::BF162,
                vectorization: self.vectorization / 2,
            },
            _ => *self,
        }
    }
}

impl<D: Dialect> Elem<D> {
    pub fn size(&self) -> usize {
        match self {
            Self::F16 => core::mem::size_of::<f16>(),
            Self::F162 => 2 * core::mem::size_of::<f16>(),
            Self::BF162 => 2 * core::mem::size_of::<bf16>(),
            Self::BF16 => core::mem::size_of::<bf16>(),
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
            Self::Atomic(AtomicKind::I32) => core::mem::size_of::<i32>(),
            Self::Atomic(AtomicKind::U32) => core::mem::size_of::<u32>(),
            Self::_Dialect(_) => 0,
        }
    }
}
