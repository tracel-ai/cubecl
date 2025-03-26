use cubecl_core::{
    ir::{self as gpu, BarrierLevel, ConstantScalarValue, Id},
    tf32,
};
use half::{bf16, f16};
use std::fmt::Display;

use super::{COUNTER_TMP_VAR, Dialect, Fragment, FragmentIdent};

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum Elem<D: Dialect> {
    TF32,
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
    U8,
    U16,
    U32,
    U64,
    Bool,
    Atomic(AtomicKind<D>),
    _Dialect(std::marker::PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum AtomicKind<D: Dialect> {
    I32,
    I64,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    /// Required to construct the inner `Elem` of the atomic value
    _Dialect(std::marker::PhantomData<D>),
}

impl<D: Dialect> Display for AtomicKind<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I32 => Elem::<D>::I32.fmt(f),
            Self::I64 => Elem::<D>::I64.fmt(f),
            Self::U32 => Elem::<D>::U32.fmt(f),
            Self::U64 => Elem::<D>::U64.fmt(f),
            Self::F16 => Elem::<D>::F16.fmt(f),
            Self::BF16 => Elem::<D>::BF16.fmt(f),
            Self::F32 => Elem::<D>::F32.fmt(f),
            Self::F64 => Elem::<D>::F64.fmt(f),
            Self::_Dialect(_) => Ok(()),
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
            Elem::F64 => f.write_str("double"),
            Elem::BF16 => D::bfloat16_type_name(f),
            Elem::BF162 => D::bfloat162_type_name(f),
            Elem::TF32 => f.write_str("float"),
            Elem::I8 => f.write_str("char"),
            Elem::I16 => f.write_str("short"),
            Elem::I32 => f.write_str("int"),
            Elem::I64 => f.write_str("int64"),
            Elem::U8 => f.write_str("uint8"),
            Elem::U16 => f.write_str("uint16"),
            Elem::U32 => f.write_str("uint"),
            Elem::U64 => f.write_str("uint64"),
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
        matches!(self.var, Variable::LocalConst { .. })
    }
}

impl<D: Dialect> Component<D> for Variable<D> {
    fn index(&self, index: usize) -> IndexedVariable<D> {
        self.index(index)
    }

    fn item(&self) -> Item<D> {
        match self {
            Variable::GlobalInputArray(_, e) => *e,
            Variable::GlobalOutputArray(_, e) => *e,
            Variable::SharedMemory(_, e, _) => *e,
            Variable::ConstantArray(_, e, _) => *e,
            Variable::LocalMut { item, .. } => *item,
            Variable::LocalConst { item, .. } => *item,
            Variable::Named { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::ConstantScalar(_, e) => Item::scalar(*e),
            Variable::GlobalScalar(_, e, _) => Item::scalar(*e),
            Variable::IdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxX => Item::scalar(Elem::U32),
            Variable::ThreadIdxY => Item::scalar(Elem::U32),
            Variable::ThreadIdxZ => Item::scalar(Elem::U32),
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
            Variable::LocalArray(_, e, _) => *e,
            Variable::WarpSize => Item::scalar(Elem::U32),
            Variable::ThreadIdxWarp => Item::scalar(Elem::U32),
            Variable::WmmaFragment { frag, .. } => Item::scalar(frag.elem),
            Variable::BlockIdxGlobal => Item::scalar(Elem::U32),
            Variable::BlockDimGlobal => Item::scalar(Elem::U32),
            Variable::GridDimGlobal => Item::scalar(Elem::U32),
            Variable::Tmp { item, .. } => *item,
            Variable::Pipeline { item, .. } => *item,
            Variable::Barrier { item, .. } => *item,
            Variable::TensorMap(_) => unreachable!(),
        }
    }

    fn is_const(&self) -> bool {
        matches!(self, Variable::LocalConst { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variable<D: Dialect> {
    WarpSize,
    ThreadIdxWarp,
    GlobalInputArray(Id, Item<D>),
    GlobalOutputArray(Id, Item<D>),
    GlobalScalar(Id, Elem<D>, gpu::Elem),
    TensorMap(Id),
    ConstantArray(Id, Item<D>, u32),
    ConstantScalar(ConstantScalarValue, Elem<D>),
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
    IdxGlobal,
    ThreadIdxGlobal,
    ThreadIdxX,
    ThreadIdxY,
    ThreadIdxZ,
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
    },
}

impl<D: Dialect> Display for Variable<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::TensorMap(id) => write!(f, "constant_{id}"),
            Variable::LocalMut { id, .. } => f.write_fmt(format_args!("l_mut_{id}")),
            Variable::LocalConst { id, .. } => f.write_fmt(format_args!("l_{id}")),
            Variable::Named { name, .. } => f.write_fmt(format_args!("{name}")),
            Variable::Slice { id, .. } => {
                write!(f, "slice_{id}")
            }
            Variable::GlobalOutputArray(number, _) => write!(f, "output_{number}"),
            Variable::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            Variable::ConstantScalar(number, elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    gpu::IntKind::I8 => write!(f, "{elem}({})", *val as i8),
                    gpu::IntKind::I16 => write!(f, "{elem}({})", *val as i16),
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
            Variable::LocalArray(id, _, _) => {
                write!(f, "l_arr_{}", id)
            }
            Variable::WarpSize => f.write_str("warpSize"),
            Variable::ThreadIdxWarp => f.write_str("threadIdxGlobal % warpSize"),
            Variable::WmmaFragment { id: index, frag } => {
                let name = match frag.ident {
                    FragmentIdent::A => "a",
                    FragmentIdent::B => "b",
                    FragmentIdent::Accumulator => "acc",
                    FragmentIdent::_Dialect(_) => "",
                };
                write!(f, "frag_{name}_{index}")
            }
            Variable::GridDimGlobal => f.write_str("gridDimGlobal"),
            Variable::Tmp { id, .. } => write!(f, "_tmp_{id}"),
            Variable::Pipeline { id, .. } => write!(f, "pipeline_{id}"),
            Variable::Barrier { id, .. } => write!(f, "barrier_{id}"),
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
            id: inc as Id,
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
            Variable::GlobalScalar(_, _, _) => true,
            Variable::ConstantScalar(_, _) => true,
            Variable::IdxGlobal => true,
            Variable::ThreadIdxGlobal => true,
            Variable::ThreadIdxX => true,
            Variable::ThreadIdxY => true,
            Variable::ThreadIdxZ => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::LocalMut { .. } => false,
            Variable::LocalConst { .. } => false,
            Variable::Named { .. } => false,
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
            Variable::LocalArray(_, _, _) => false,
            Variable::WarpSize => true,
            Variable::ThreadIdxWarp => true,
            Variable::WmmaFragment { .. } => false,
            Variable::BlockIdxGlobal => true,
            Variable::BlockDimGlobal => true,
            Variable::GridDimGlobal => true,
            Variable::Tmp { .. } => false,
            Variable::Pipeline { .. } => false,
            Variable::Barrier { .. } => false,
            Variable::TensorMap(_) => false,
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
            Variable::GlobalScalar(id, ..) => Some(*id),
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
}

pub trait FmtLeft: Display {
    fn fmt_left(&self) -> String;
}

impl<D: Dialect> FmtLeft for Variable<D> {
    fn fmt_left(&self) -> String {
        match self {
            Self::LocalConst { item, .. } => format!("const {item} {self}"),
            Variable::Tmp { item, .. } => format!("{item} {self}"),
            var => format!("{var}"),
        }
    }
}

impl<D: Dialect> FmtLeft for IndexedVariable<D> {
    fn fmt_left(&self) -> String {
        match self.var {
            Variable::LocalConst { item, .. } => format!("const {item} {self}"),
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
        let ref_ = matches!(var, Variable::LocalConst { .. })
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
    pub const fn size(&self) -> usize {
        match self {
            Elem::F16 => core::mem::size_of::<f16>(),
            Elem::F162 => 2 * core::mem::size_of::<f16>(),
            Elem::BF162 => 2 * core::mem::size_of::<bf16>(),
            Elem::BF16 => core::mem::size_of::<bf16>(),
            Elem::TF32 => core::mem::size_of::<tf32>(),
            Elem::F32 => core::mem::size_of::<f32>(),
            Elem::F64 => core::mem::size_of::<f64>(),
            Elem::I8 => core::mem::size_of::<i8>(),
            Elem::I16 => core::mem::size_of::<i16>(),
            Elem::I32 => core::mem::size_of::<i32>(),
            Elem::I64 => core::mem::size_of::<i64>(),
            Elem::U8 => core::mem::size_of::<u8>(),
            Elem::U16 => core::mem::size_of::<u16>(),
            Elem::U32 => core::mem::size_of::<u32>(),
            Elem::U64 => core::mem::size_of::<u64>(),
            Elem::Bool => core::mem::size_of::<bool>(),
            Elem::Atomic(AtomicKind::I32) => core::mem::size_of::<i32>(),
            Elem::Atomic(AtomicKind::I64) => core::mem::size_of::<i64>(),
            Elem::Atomic(AtomicKind::U32) => core::mem::size_of::<u32>(),
            Elem::Atomic(AtomicKind::U64) => core::mem::size_of::<u64>(),
            Elem::Atomic(AtomicKind::F16) => core::mem::size_of::<f16>(),
            Elem::Atomic(AtomicKind::BF16) => core::mem::size_of::<bf16>(),
            Elem::Atomic(AtomicKind::F32) => core::mem::size_of::<f32>(),
            Elem::Atomic(AtomicKind::F64) => core::mem::size_of::<f64>(),
            Elem::Atomic(AtomicKind::_Dialect(_)) => 0,
            Elem::_Dialect(_) => 0,
        }
    }
}
