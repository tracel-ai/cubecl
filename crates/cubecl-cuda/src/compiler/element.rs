use cubecl_core::ir::{self as gpu, ConstantScalarValue};
use half::{bf16, f16};
use std::fmt::Display;

use super::Fragment;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    F16,
    BF16,
    I32,
    U32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::F16 => f.write_str("__half"),
            Elem::F32 => f.write_str("float"),
            Elem::BF16 => f.write_str("__nv_bfloat16"),
            Elem::I32 => f.write_str("int"),
            Elem::U32 => f.write_str("uint"),
            Elem::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Vec4(elem) => match elem {
                Elem::F32 => f.write_str("float4"),
                Elem::I32 => f.write_str("int4"),
                Elem::U32 => f.write_str("uint4"),
                Elem::Bool => f.write_str("bool4"),
                Elem::BF16 => f.write_str("__nv_bfloat164"),
                Elem::F16 => f.write_str("half4"),
            },
            Item::Vec3(elem) => match elem {
                Elem::F32 => f.write_str("float3"),
                Elem::I32 => f.write_str("int3"),
                Elem::U32 => f.write_str("uint3"),
                Elem::Bool => f.write_str("bool3"),
                Elem::BF16 => f.write_str("__nv_bfloat164"),
                Elem::F16 => f.write_str("half3"),
            },
            Item::Vec2(elem) => match elem {
                Elem::F32 => f.write_str("float2"),
                Elem::I32 => f.write_str("int2"),
                Elem::U32 => f.write_str("uint2"),
                Elem::Bool => f.write_str("bool2"),
                Elem::BF16 => f.write_str("__nv_bfloat162"),
                Elem::F16 => f.write_str("half2"),
            },
            Item::Scalar(elem) => f.write_fmt(format_args!("{elem}")),
        }
    }
}

pub trait Component: Display {
    fn item(&self) -> Item;
    fn elem(&self) -> Elem {
        *self.item().elem()
    }
}

impl Component for IndexedVariable {
    fn item(&self) -> Item {
        self.var.item()
    }
}
impl Component for Variable {
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
            Variable::ConstantScalar(_, e) => Item::Scalar(*e),
            Variable::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Variable::IdxGlobal => Item::Scalar(Elem::U32),
            Variable::ThreadIdxGlobal => Item::Scalar(Elem::U32),
            Variable::ThreadIdxX => Item::Scalar(Elem::U32),
            Variable::ThreadIdxY => Item::Scalar(Elem::U32),
            Variable::ThreadIdxZ => Item::Scalar(Elem::U32),
            Variable::Rank => Item::Scalar(Elem::U32),
            Variable::LocalScalar {
                id: _,
                elem,
                depth: _,
            } => Item::Scalar(*elem),
            Variable::BlockIdxX => Item::Scalar(Elem::U32),
            Variable::BlockIdxY => Item::Scalar(Elem::U32),
            Variable::BlockIdxZ => Item::Scalar(Elem::U32),
            Variable::AbsoluteIdxX => Item::Scalar(Elem::U32),
            Variable::AbsoluteIdxY => Item::Scalar(Elem::U32),
            Variable::AbsoluteIdxZ => Item::Scalar(Elem::U32),
            Variable::BlockDimX => Item::Scalar(Elem::U32),
            Variable::BlockDimY => Item::Scalar(Elem::U32),
            Variable::BlockDimZ => Item::Scalar(Elem::U32),
            Variable::GridDimX => Item::Scalar(Elem::U32),
            Variable::GridDimY => Item::Scalar(Elem::U32),
            Variable::GridDimZ => Item::Scalar(Elem::U32),
            Variable::LocalArray(_, e, _, _) => *e,
            Variable::WarpSize => Item::Scalar(Elem::U32),
            Variable::WmmaFragment { id: _, frag } => Item::Scalar(frag.elem),
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
    BlockIdxX,
    BlockIdxY,
    BlockIdxZ,
    AbsoluteIdxX,
    AbsoluteIdxY,
    AbsoluteIdxZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimX,
    GridDimY,
    GridDimZ,
    WmmaFragment { id: u16, frag: Fragment },
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
            Variable::BlockIdxX => f.write_str("blockIdx.x"),
            Variable::BlockIdxY => f.write_str("blockIdx.y"),
            Variable::BlockIdxZ => f.write_str("blockIdx.z"),
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
            Variable::WmmaFragment { id: index, frag: _ } => {
                f.write_fmt(format_args!("frag_{index}"))
            }
        }
    }
}

impl Variable {
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
            Variable::WmmaFragment { id: _, frag: _ } => false,
        }
    }

    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable { var: *self, index }
    }
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let item = self.var.item();

        match item {
            Item::Vec4(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                2 => f.write_fmt(format_args!("{var}.z"))?,
                3 => f.write_fmt(format_args!("{var}.w"))?,
                _ => unreachable!(),
            },
            Item::Vec3(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                2 => f.write_fmt(format_args!("{var}.z"))?,
                _ => unreachable!(),
            },
            Item::Vec2(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                _ => unreachable!(),
            },
            Item::Scalar(_) => f.write_fmt(format_args!("{var}"))?,
        }

        Ok(())
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
}

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::F16 => core::mem::size_of::<f16>(),
            Self::BF16 => core::mem::size_of::<bf16>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}
