use cubecl_core::ir::{self as gpu, ConstantScalarValue};
use half::{bf16, f16};
use std::fmt::Display;

use super::Fragment;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum Elem {
    F32,
    F16,
    F162,
    BF16,
    BF162,
    I32,
    U32,
    Bool,
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
            Elem::BF16 => f.write_str("__nv_bfloat16"),
            Elem::BF162 => f.write_str("__nv_bfloat162"),
            Elem::I32 => f.write_str("int"),
            Elem::U32 => f.write_str("uint"),
            Elem::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if 1 == self.vectorization {
            return f.write_fmt(format_args!("{}", self.elem));
        }

        return f.write_fmt(format_args!("{}_{}", self.elem, self.vectorization));
    }
}

pub trait Component: Display {
    fn item(&self) -> Item;
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
            Variable::Local {
                id: _,
                item,
                depth: _,
            } => *item,
            Variable::Slice {
                id: _,
                item,
                depth: _,
            } => *item,
            Variable::ConstantScalar(_, e) => Item::scalar(*e),
            Variable::GlobalScalar(_, e, _) => Item::scalar(*e),
            Variable::IdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxGlobal => Item::scalar(Elem::U32),
            Variable::ThreadIdxX => Item::scalar(Elem::U32),
            Variable::ThreadIdxY => Item::scalar(Elem::U32),
            Variable::ThreadIdxZ => Item::scalar(Elem::U32),
            Variable::Rank => Item::scalar(Elem::U32),
            Variable::LocalScalar {
                id: _,
                elem,
                depth: _,
            } => Item::scalar(*elem),
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
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Variable {
    WarpSize,
    GlobalInputArray(u16, Item),
    GlobalOutputArray(u16, Item),
    GlobalScalar(u16, Elem, gpu::Elem),
    ConstantScalar(ConstantScalarValue, Elem),
    Local { id: u16, item: Item, depth: u8 },
    Slice { id: u16, item: Item, depth: u8 },
    LocalScalar { id: u16, elem: Elem, depth: u8 },
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
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::LocalScalar {
                id: index,
                elem: _,
                depth: scope_depth,
            } => f.write_fmt(format_args!("s_{scope_depth}_{index}")),
            Variable::Local {
                id: index,
                item: _,
                depth: scope_depth,
            } => f.write_fmt(format_args!("l_{scope_depth}_{index}")),
            Variable::Slice { id, item: _, depth } => {
                f.write_fmt(format_args!("slice_{depth}_{id}"))
            }
            Variable::GlobalOutputArray(number, _) => f.write_fmt(format_args!("output_{number}")),
            Variable::GlobalScalar(number, _, elem) => {
                f.write_fmt(format_args!("scalars_{elem}[{number}]"))
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Variable::ConstantScalar(number, elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    gpu::IntKind::I32 => f.write_fmt(format_args!("{elem}({})", *val as i32)),
                    gpu::IntKind::I64 => f.write_fmt(format_args!("{elem}({})", { *val })),
                },
                ConstantScalarValue::Float(val, kind) => match kind {
                    gpu::FloatKind::F16 => {
                        f.write_fmt(format_args!("{elem}({:?})", half::f16::from_f64(*val)))
                    }
                    gpu::FloatKind::BF16 => {
                        f.write_fmt(format_args!("{elem}({:?})", half::bf16::from_f64(*val)))
                    }
                    gpu::FloatKind::F32 => f.write_fmt(format_args!("{elem}({:?})", *val as f32)),
                    gpu::FloatKind::F64 => f.write_fmt(format_args!("{elem}({:?})", { *val })),
                },
                ConstantScalarValue::UInt(val) => {
                    f.write_fmt(format_args!("{elem}({})", *val as u32))
                }
                ConstantScalarValue::Bool(val) => f.write_fmt(format_args!("{}", val)),
            },
            Variable::SharedMemory(number, _, _) => {
                f.write_fmt(format_args!("shared_memory_{number}"))
            }
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
                f.write_fmt(format_args!("l_arr_{}_{}", id, depth))
            }
            Variable::WarpSize => f.write_str("warpSize"),
            Variable::WmmaFragment {
                id: index,
                frag: _,
                depth,
            } => f.write_fmt(format_args!("frag_{index}_{depth}")),
            Variable::GridDimGlobal => f.write_str("gridDimGlobal"),
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
            Variable::LocalScalar {
                id: _,
                elem: _,
                depth: _,
            } => true,
            Variable::IdxGlobal => true,
            Variable::ThreadIdxGlobal => true,
            Variable::ThreadIdxX => true,
            Variable::ThreadIdxY => true,
            Variable::ThreadIdxZ => true,
            Variable::Rank => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::Local {
                id: _,
                item: _,
                depth: _,
            } => false,
            Variable::Slice {
                id: _,
                item: _,
                depth: _,
            } => false,
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
            Variable::WmmaFragment {
                id: _,
                frag: _,
                depth: _,
            } => false,
            Variable::BlockIdxGlobal => true,
            Variable::BlockDimGlobal => true,
            Variable::GridDimGlobal => true,
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

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    optimized: bool,
    index: usize,
}

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;

        if self.var.item().vectorization > 1 {
            if self.optimized {
                let item = self.var.item();
                f.write_fmt(format_args!(
                    "(reinterpret_cast<{item}&>({var})).i_{}",
                    self.index
                ))
            } else {
                f.write_fmt(format_args!("{var}.i_{}", self.index))
            }
        } else {
            if self.optimized {
                let item = self.var.item();
                f.write_fmt(format_args!("reinterpret_cast<{item}&>({var})"))
            } else {
                f.write_fmt(format_args!("{var}"))
            }
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
            Self::F16 => core::mem::size_of::<f16>(),
            Self::F162 => 2 * core::mem::size_of::<f16>(),
            Self::BF162 => 2 * core::mem::size_of::<bf16>(),
            Self::BF16 => core::mem::size_of::<bf16>(),
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}
