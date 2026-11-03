//! What one plane contracts through ([`Instruction`]).

use cubecl::cmma::MatrixIdent;
use cubecl::ir::{ElemType, FloatKind, features::ScaledMmaConfig};

use crate::{MmaIo, RegisterBlock};

/// What one plane contracts through: a software register block or a hardware fragment form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    /// A register block, run under `config`. Asks nothing of the hardware.
    Registers { config: RegisterBlock },
    /// The vendor's fragment API: `wmma` on CUDA, rocWMMA on HIP.
    Cmma,
    /// This crate's own fragment transport over the matrix intrinsics, with per-role `io`, over
    /// `operands` of one form.
    Mma { io: MmaIo, operands: MmaOperands },
    /// Hopper's warpgroup MMA: a plane group of four contracts its `64 × n` tile together,
    /// reading both operands out of shared memory, asynchronously.
    Wgmma,
}

impl Instruction {
    /// Whether a sum this instruction contracts is scaled where it is held: a register block's
    /// cells and a manual-mma fragment's are each unit's own registers. A cmma fragment hides
    /// which unit holds a cell, so it is scaled only through a round trip in shared memory, and a
    /// warpgroup's sum only once its MMAs land.
    pub(crate) fn scales_its_sum_in_place(self) -> bool {
        match self {
            Self::Registers { .. } | Self::Mma { .. } => true,
            Self::Cmma | Self::Wgmma => false,
        }
    }
}

/// What a manual-mma instruction's operands are: the blueprint's choice, which every leaf reads, so
/// the operands a leaf is handed never decide it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MmaOperands {
    /// Values as the instruction takes them. A scaled or packed operand is decoded before it
    /// reaches one, or read through lane runs where its fragment is deeper than the instruction.
    #[default]
    Plain,
    /// Packed codes under block scales, each read as stored by the device's block-scaled
    /// instruction of this form.
    BlockScaled(BlockScaledOperands),
}

impl MmaOperands {
    /// The block-scaled instruction's form, where the operands are read by one.
    pub fn block_scaled(self) -> Option<BlockScaledOperands> {
        match self {
            MmaOperands::Plain => None,
            MmaOperands::BlockScaled(form) => Some(form),
        }
    }
}

/// The form of a block-scaled instruction's operands: what each input's registers hold, what its
/// scales are, and how many values of `k` one scale covers. Any form the device's instruction
/// takes ([`ScaledMmaConfig`]); the inputs may hold different codes, under scales of one type and
/// one block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockScaledOperands {
    /// The `A`'s codes as a register of the instruction holds them: `e2m1x2` for `e2m1` codes two
    /// to a byte, `e4m3` for one to a byte.
    pub lhs: FloatKind,
    /// The `B`'s codes, as [`lhs`](Self::lhs) is the `A`'s.
    pub rhs: FloatKind,
    /// The type of a block scale.
    pub scales: FloatKind,
    /// Values of `k` one scale covers.
    pub block: usize,
}

impl BlockScaledOperands {
    /// The device instruction of `m × n × k` this form names, summing in `f32`.
    pub fn instruction(&self, m: usize, n: usize, k: usize) -> ScaledMmaConfig {
        ScaledMmaConfig {
            a_type: ElemType::Float(self.lhs),
            b_type: ElemType::Float(self.rhs),
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(self.scales),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: (k / self.block) as u32,
        }
    }

    /// The form of `instruction`, a device's block-scaled instruction.
    pub fn of(instruction: &ScaledMmaConfig) -> Option<Self> {
        let float = |elem: ElemType| match elem {
            ElemType::Float(kind) => Some(kind),
            _ => None,
        };
        let scales = instruction.scales_factor as usize;
        Some(Self {
            lhs: float(instruction.a_type)?,
            rhs: float(instruction.b_type)?,
            scales: float(instruction.scales_type)?,
            block: (instruction.k as usize).checked_div(scales)?,
        })
    }

    /// The codes the operand in role `ident` holds.
    pub fn codes(&self, ident: MatrixIdent) -> FloatKind {
        match ident {
            MatrixIdent::A => self.lhs,
            _ => self.rhs,
        }
    }

    /// The codes a register of the instruction holds in role `lhs` (or `rhs`), counted in values:
    /// two for each `e2m1x2`.
    pub fn values_per_element(kind: FloatKind) -> usize {
        match kind {
            FloatKind::E2M1x2 => 2,
            _ => 1,
        }
    }
}
