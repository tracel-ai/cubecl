use cubecl_core::ir::{self as gpu, ConstantScalarValue};
use half::{bf16, f16};
use std::fmt::Display;

use super::{Fragment, COUNTER_TMP_VAR};

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum Elem {
    F32,
    F64,
    F16,
    F162,
    BF16,
    BF162,
    I8,
    I16,
    I32,
    I64,
    U32,
    Bool,
    Atomic(AtomicKind),
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
pub struct Item {
    pub(crate) elem: Elem,
    pub(crate) vectorization: usize,
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::F16 => f.write_str("__half"),
            Elem::F162 => f.write_str("__half2"),
            Elem::F32 => f.write_str("float"),
            Elem::F64 => f.write_str("double"),
            Elem::BF16 => f.write_str("__nv_bfloat16"),
            Elem::BF162 => f.write_str("__nv_bfloat162"),
            Elem::I8 => f.write_str("char"),
            Elem::I16 => f.write_str("short"),
            Elem::I32 => f.write_str("int"),
            Elem::I64 => f.write_str("int64"),
            Elem::U32 => f.write_str("uint"),
            Elem::Bool => f.write_str("bool"),
            Elem::Atomic(inner) => inner.fmt(f),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if 1 == self.vectorization {
            return write!(f, "{}", self.elem);
        }

        write!(f, "{}_{}", self.elem, self.vectorization)
    }
}

pub trait Component: Display + FmtLeft {
    fn item(&self) -> Item;
    fn is_const(&self) -> bool;
    fn index(&self, index: usize) -> IndexedVariable;
    fn elem(&self) -> Elem {
        *self.item().elem()
    }
}

impl Component for IndexedVariable {
    fn item(&self) -> Item {
        self.var.item()
    }

    fn index(&self, index: usize) -> IndexedVariable {
        self.var.index(index)
    }

    fn is_const(&self) -> bool {
        matches!(self.var, Variable::ConstLocal { .. })
    }
}
impl Component for Variable {
    fn index(&self, index: usize) -> IndexedVariable {
        self.index(index)
    }

    fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, e) => *e,
            Variable::GlobalOutputArray(_, e) => *e,
            Variable::SharedMemory(_, e, _) => *e,
            Variable::ConstantArray(_, e, _) => *e,
            Variable::Local { item, .. } => *item,
            Variable::ConstLocal { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::ConstantScalar(_, e) => Item::scalar(*e),
            Variable::GlobalScalar(_, e, _) => Item::scalar(*e),
            Variable::IdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxX => Item::scalar(Elem::U32),
            Variable::ThreadIdxY => Item::scalar(Elem::U32),
            Variable::ThreadIdxZ => Item::scalar(Elem::U32),
            Variable::Rank => Item::scalar(Elem::U32),
            Variable::BlockIdxX => Item::scalar(Elem::U32),
            Variable::BlockIdxY => Item::scalar(Elem::U32),
            Variable::BlockIdxZ => Item::scalar(Elem::U32),
            Variable::AbsoluteIdxX => Item::scalar(Elem::U32),
            Variable::AbsoluteIdxY => Item::scalar(Elem::U32),
            Variable::AbsoluteIdxZ => Item::scalar(Elem::U32),
            Variable::BlockDimX => Item::scalar(Elem::U32),
            Variable::BlockDimY => Item::scalar(Elem::U32),
            Variable::BlockDimZ => Item::scalar(Elem::U32),
            Variable::GridDimX => Item::scalar(Elem::U32),
            Variable::GridDimY => Item::scalar(Elem::U32),
            Variable::GridDimZ => Item::scalar(Elem::U32),
            Variable::LocalArray(_, e, _, _) => *e,
            Variable::WarpSize => Item::scalar(Elem::U32),
            Variable::WmmaFragment {
                id: _,
                frag,
                depth: _,
            } => Item::scalar(frag.elem),
            Variable::BlockIdxGlobal => Item::scalar(Elem::U32),
            Variable::BlockDimGlobal => Item::scalar(Elem::U32),
            Variable::GridDimGlobal => Item::scalar(Elem::U32),
            Variable::Tmp { item, .. } => *item,
        }
    }

    fn is_const(&self) -> bool {
        matches!(self, Variable::ConstLocal { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variable {
    WarpSize,
    GlobalInputArray(u16, Item),
    GlobalOutputArray(u16, Item),
    GlobalScalar(u16, Elem, gpu::Elem),
    ConstantArray(u16, Item, u32),
    ConstantScalar(ConstantScalarValue, Elem),
    Local { id: u16, item: Item, depth: u8 },
    ConstLocal { id: u16, item: Item, depth: u8 },
    Slice { id: u16, item: Item, depth: u8 },
    SharedMemory(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
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
    WmmaFragment { id: u16, frag: Fragment, depth: u8 },
    Tmp { id: u16, item: Item },
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::Local { id, depth, .. } => f.write_fmt(format_args!("l_{depth}_{id}")),
            Variable::ConstLocal { id, depth, .. } => f.write_fmt(format_args!("ssa_{depth}_{id}")),
            Variable::Slice { id, item: _, depth } => {
                write!(f, "slice_{depth}_{id}")
            }
            Variable::GlobalOutputArray(number, _) => write!(f, "output_{number}"),
            Variable::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Variable::ConstantScalar(number, elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    gpu::IntKind::I8 => write!(f, "{elem}({})", *val as i32),
                    gpu::IntKind::I16 => write!(f, "{elem}({})", *val as i32),
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
            Variable::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }
            Variable::ConstantArray(number, _, _) => f.write_fmt(format_args!("arrays_{number}")),
            Variable::ThreadIdxGlobal => f.write_str("threadIdxGlobal"),
            Variable::ThreadIdxX => f.write_str("threadIdx.x"),
            Variable::ThreadIdxY => f.write_str("threadIdx.y"),
            Variable::ThreadIdxZ => f.write_str("threadIdx.z"),
            Variable::Rank => f.write_str("rank"),
            Variable::BlockIdxGlobal => f.write_str("blockIdxGlobal"),
            Variable::BlockIdxX => f.write_str("blockIdx.x"),
            Variable::BlockIdxY => f.write_str("blockIdx.y"),
            Variable::BlockIdxZ => f.write_str("blockIdx.z"),
            Variable::BlockDimGlobal => f.write_str("blockDimGlobal"),
            Variable::BlockDimX => f.write_str("blockDim.x"),
            Variable::BlockDimY => f.write_str("blockDim.y"),
            Variable::BlockDimZ => f.write_str("blockDim.z"),
            Variable::IdxGlobal => f.write_str("idxGlobal"),
            Variable::GridDimX => f.write_str("gridDim.x"),
            Variable::GridDimY => f.write_str("gridDim.y"),
            Variable::GridDimZ => f.write_str("gridDim.z"),
            Variable::AbsoluteIdxX => f.write_str("absoluteIdx.x"),
            Variable::AbsoluteIdxY => f.write_str("absoluteIdx.y"),
            Variable::AbsoluteIdxZ => f.write_str("absoluteIdx.z"),
            Variable::LocalArray(id, _item, depth, _size) => {
                write!(f, "l_arr_{}_{}", id, depth)
            }
            Variable::WarpSize => f.write_str("warpSize"),
            Variable::WmmaFragment {
                id: index,
                frag: _,
                depth,
            } => write!(f, "frag_{index}_{depth}"),
            Variable::GridDimGlobal => f.write_str("gridDimGlobal"),
            Self::Tmp { id, .. } => write!(f, "_tmp_{id}"),
        }
    }
}

#[derive(new)]
pub struct OptimizedArgs<const N: usize> {
    pub args: [Variable; N],
    pub optimization_factor: Option<usize>,
}

impl Variable {
    pub fn is_optimized(&self) -> bool {
        self.item().is_optimized()
    }

    pub fn tmp(item: Item) -> Self {
        let inc = COUNTER_TMP_VAR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Variable::Tmp {
            id: inc as u16,
            item,
        }
    }

    pub fn optimized_args<const N: usize>(args: [Self; N]) -> OptimizedArgs<N> {
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
            Variable::Local { id, item, depth } => Variable::Local {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Variable::ConstLocal { id, item, depth } => Variable::ConstLocal {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Variable::Slice { id, item, depth } => Variable::Slice {
                id: *id,
                item: item.optimized(),
                depth: *depth,
            },
            Variable::SharedMemory(id, item, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Variable::SharedMemory(*id, item, size / scaling)
            }
            Variable::LocalArray(id, item, vec, size) => {
                let before = item.vectorization;
                let item = item.optimized();
                let after = item.vectorization;
                let scaling = (before / after) as u32;

                Variable::LocalArray(*id, item.optimized(), *vec, size / scaling)
            }
            _ => *self,
        }
    }

    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::GlobalScalar(_, _, _) => true,
            Variable::ConstantScalar(_, _) => true,
            Variable::IdxGlobal => true,
            Variable::ThreadIdxGlobal => true,
            Variable::ThreadIdxX => true,
            Variable::ThreadIdxY => true,
            Variable::ThreadIdxZ => true,
            Variable::Rank => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::Local { .. } => false,
            Variable::ConstLocal { .. } => false,
            Variable::Slice { .. } => false,
            Variable::BlockIdxX => true,
            Variable::BlockIdxY => true,
            Variable::BlockIdxZ => true,
            Variable::AbsoluteIdxX => true,
            Variable::AbsoluteIdxY => true,
            Variable::AbsoluteIdxZ => true,
            Variable::BlockDimX => true,
            Variable::BlockDimY => true,
            Variable::BlockDimZ => true,
            Variable::GridDimX => true,
            Variable::GridDimY => true,
            Variable::GridDimZ => true,
            Variable::LocalArray(_, _, _, _) => false,
            Variable::WarpSize => true,
            Variable::WmmaFragment { .. } => false,
            Variable::BlockIdxGlobal => true,
            Variable::BlockDimGlobal => true,
            Variable::GridDimGlobal => true,
            Variable::Tmp { .. } => false,
        }
    }

    pub fn index(&self, index: usize) -> IndexedVariable {
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

impl FmtLeft for Variable {
    fn fmt_left(&self) -> String {
        match self {
            Self::ConstLocal { item, .. } => format!("const {item} {self}"),
            Variable::Tmp { item, .. } => format!("{item} {self}"),
            var => format!("{var}"),
        }
    }
}

impl FmtLeft for IndexedVariable {
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
pub struct IndexedVariable {
    var: Variable,
    optimized: bool,
    index: usize,
}

impl Display for IndexedVariable {
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
impl Item {
    pub fn elem(&self) -> &Elem {
        &self.elem
    }

    pub fn de_optimized(&self) -> Self {
        match self.elem {
            Elem::F162 => Item::new(Elem::F16, self.vectorization * 2),
            Elem::BF162 => Item::new(Elem::BF16, self.vectorization * 2),
            _ => *self,
        }
    }

    pub fn new(elem: Elem, vectorization: usize) -> Self {
        Self {
            elem,
            vectorization,
        }
    }
    pub fn scalar(elem: Elem) -> Self {
        Self {
            elem,
            vectorization: 1,
        }
    }

    pub fn is_optimized(&self) -> bool {
        matches!(self.elem, Elem::F162 | Elem::BF162)
    }

    pub fn optimized(&self) -> Item {
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

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Elem::F16 => core::mem::size_of::<f16>(),
            Elem::F162 => 2 * core::mem::size_of::<f16>(),
            Elem::BF162 => 2 * core::mem::size_of::<bf16>(),
            Elem::BF16 => core::mem::size_of::<bf16>(),
            Elem::F32 => core::mem::size_of::<f32>(),
            Elem::F64 => core::mem::size_of::<f64>(),
            Elem::I8 => core::mem::size_of::<i8>(),
            Elem::I16 => core::mem::size_of::<i16>(),
            Elem::I32 => core::mem::size_of::<i32>(),
            Elem::I64 => core::mem::size_of::<i64>(),
            Elem::U32 => core::mem::size_of::<u32>(),
            Elem::Bool => core::mem::size_of::<bool>(),
            Elem::Atomic(AtomicKind::I32) => core::mem::size_of::<i32>(),
            Elem::Atomic(AtomicKind::U32) => core::mem::size_of::<u32>(),
        }
    }
}
