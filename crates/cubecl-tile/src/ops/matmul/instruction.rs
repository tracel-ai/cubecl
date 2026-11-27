//! What one plane contracts through ([`Instruction`]).

use cubecl::cmma::{MatrixIdent, MatrixShape};
use cubecl::ir::features::{MmaConfig, Plane};
use cubecl::ir::{
    DeviceProperties, ElemType, FloatKind, TargetProperties, features::ScaledMmaConfig,
};

use cubecl::quant::scheme::QuantValue;

use crate::ops::matmul::mma_io::LaneVectors;
use crate::tile::kind::plane::mma::lane_runs::LaneMap;
use crate::tile::packing::base::Field;
use crate::{ComputeScope, KeyOrder, MmaIo, SoftwareMatmul};

/// What one plane contracts through: a software matmul or a hardware fragment form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    /// A software matmul, run under `config`. Asks nothing of the hardware.
    Registers { config: SoftwareMatmul },
    /// The vendor's fragment API: `wmma` on CUDA, rocWMMA on HIP.
    Cmma,
    /// This crate's own fragment transport over the matrix intrinsics, with per-role `io`, over
    /// `operands` of one form.
    Mma { io: MmaIo, operands: MmaOperands },
    /// Hopper's plane group MMA: a plane group of four contracts its `64 × n` tile together,
    /// reading both operands out of shared memory, asynchronously.
    Wgmma,
}

impl Instruction {
    /// This instruction with its `B` reading its rows in `keys` order: a manual-mma one's
    /// ([`MmaIo::rhs_keys`]); any other reads them as they lie.
    pub(crate) fn reading_keys(self, keys: KeyOrder) -> Self {
        match self {
            Self::Mma { io, operands } => Self::Mma {
                io: MmaIo {
                    rhs_keys: keys,
                    ..io
                },
                operands,
            },
            other => other,
        }
    }

    /// Whether a sum this instruction contracts is scaled where it is held: a register block's
    /// cells and a manual-mma fragment's are each unit's own registers. A cmma fragment hides
    /// which unit holds a cell, so it is scaled only through a round trip in shared memory, and a
    /// plane group's sum only once its MMAs land.
    pub fn scales_its_sum_in_place(self) -> bool {
        match self {
            Self::Registers { .. } | Self::Mma { .. } => true,
            Self::Cmma | Self::Wgmma => false,
        }
    }

    /// Whether a sum this instruction contracts hides which unit holds which of its cells: a cmma
    /// fragment's does, so it reaches a destination that adds, overhangs or casts out of place
    /// through a scratch, a cell at a time, and the host sizes that scratch.
    pub fn hides_cell_positions(self) -> bool {
        match self {
            Self::Cmma => true,
            Self::Registers { .. } | Self::Mma { .. } | Self::Wgmma => false,
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

/// What a plane's fragments of one operand load from, as the tile layer sees it while the kernel
/// expands: the facts [`Instruction::for_accumulator`] and
/// [`Instruction::for_stage_accumulator`] read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FragmentSource {
    /// Values of this element, held as they are or decoded by the fill that staged them.
    Values(ElemType),
    /// Values of this element the TMA engine lands: a box holds no padding, so its rows are
    /// staged swizzled, which this crate's transport reads on distinct banks.
    Landed(ElemType),
    /// Packed codes staged as they are stored, their block scales kept beside them
    /// ([`Tile::scales_kept_beside`](crate::Tile::scales_kept_beside)): read as stored by a
    /// block-scaled instruction, or decoded as each fragment loads, in lane runs.
    Codes {
        code: QuantValue,
        scales: Option<Field>,
    },
    /// Packed codes staged as they are stored, under no scale the stage keeps: decoded as each
    /// fragment loads, in lane runs.
    StoredCodes(QuantValue),
    /// Packed values a stage fill decodes on their way into the fragments.
    Decoded,
    /// A product already held in fragments of this instruction, each an `m × n` cell of it,
    /// contracted again as it lies.
    Fragments {
        instruction: Instruction,
        cell: [usize; 2],
    },
}

impl FragmentSource {
    /// The element fragments of this operand contract at, against `other`: its own, or the other
    /// operand's for packed values decoded on the way or codes decoded as they load. `None` where
    /// neither operand names one.
    pub(crate) fn element(self, other: FragmentSource) -> Option<ElemType> {
        let named = |source: FragmentSource| match source {
            FragmentSource::Values(element) | FragmentSource::Landed(element) => Some(element),
            FragmentSource::Codes { .. }
            | FragmentSource::StoredCodes(_)
            | FragmentSource::Decoded
            | FragmentSource::Fragments { .. } => None,
        };
        match self {
            FragmentSource::Decoded
            | FragmentSource::Codes { .. }
            | FragmentSource::StoredCodes(_) => named(other),
            _ => named(self),
        }
    }
}

/// Which accumulator constructor opened a plane-held sum: the one the kernel calls says what
/// becomes of the sum, so the rule never needs telling.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum OpenedBy {
    /// [`Tile::accumulator`](crate::Tile::accumulator): a sum drained into an output.
    Accumulator,
    /// [`Tile::stage_accumulator`](crate::Tile::stage_accumulator) and its sibling: a sum
    /// contracted again as the next product's `A`.
    StageAccumulator,
    /// [`Tile::rescaled_accumulator`](crate::Tile::rescaled_accumulator): a sum drained into an
    /// output whose rows are scaled between its products, as an online softmax's output is.
    RescaledAccumulator,
    /// [`Tile::held_factor_accumulator`](crate::Tile::held_factor_accumulator): a
    /// [`StageAccumulator`](Self::StageAccumulator) of products whose first factor the planes hold
    /// in registers.
    HeldFactorAccumulator,
}

impl Instruction {
    /// What the sum [`Tile::accumulator`](crate::Tile::accumulator) opens contracts through on
    /// `device` compiling for `target`, at the shape of one of its calls: a sum in `sum` held by
    /// `holder`, of fragments loaded from `lhs` and `rhs`, over a leaf of `m × n × k`. The rule the
    /// kernel reads as it expands; a host reads it to size what the kernel will hold.
    /// `None` where the device offers no matrix instruction for the product, which a plane routine
    /// refuses.
    pub fn for_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        holder: ComputeScope,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<ShapedInstruction> {
        Self::for_plane(
            device,
            target,
            holder,
            lhs,
            rhs,
            sum,
            shape,
            OpenedBy::Accumulator,
        )
    }

    /// What the sum [`Tile::stage_accumulator`](crate::Tile::stage_accumulator) opens contracts
    /// through, a sum contracted again as the next product's `A`, as an attention's score is:
    /// [`for_accumulator`](Self::for_accumulator)'s rule, except that this crate's transport comes
    /// first where `target` lays a fragment's `A` like its accumulator, so the sum lies where that
    /// `A` does. The tile cuts and walks its leaf as it does any sum's: the next product reads runs
    /// of its cells along the contraction as its `A` ([`FragmentSource::Fragments`]).
    pub fn for_stage_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<ShapedInstruction> {
        Self::for_plane(
            device,
            target,
            ComputeScope::Plane,
            lhs,
            rhs,
            sum,
            shape,
            OpenedBy::StageAccumulator,
        )
    }

    /// What the sum [`Tile::held_factor_accumulator`](crate::Tile::held_factor_accumulator) opens
    /// contracts through: [`for_stage_accumulator`](Self::for_stage_accumulator)'s rule through this
    /// crate's transport alone, which is what loads a factor once for every product. `None` where
    /// the device offers no call of it.
    pub fn for_held_factor_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<ShapedInstruction> {
        Self::for_plane(
            device,
            target,
            ComputeScope::Plane,
            lhs,
            rhs,
            sum,
            shape,
            OpenedBy::HeldFactorAccumulator,
        )
    }

    /// What the sum [`Tile::rescaled_accumulator`](crate::Tile::rescaled_accumulator) opens
    /// contracts through, a sum whose rows are scaled between its products:
    /// [`for_accumulator`](Self::for_accumulator)'s rule, except that this crate's transport comes
    /// first where it offers the call the fragment API would take. Its units hold their cells
    /// where the rows can be read, so a row is scaled in their registers; a fragment API's cells
    /// lie where only the instruction knows, and each scale bounces the sum through shared memory.
    pub fn for_rescaled_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<ShapedInstruction> {
        Self::for_plane(
            device,
            target,
            ComputeScope::Plane,
            lhs,
            rhs,
            sum,
            shape,
            OpenedBy::RescaledAccumulator,
        )
    }

    /// The rule behind [`for_accumulator`](Self::for_accumulator),
    /// [`for_stage_accumulator`](Self::for_stage_accumulator),
    /// [`for_held_factor_accumulator`](Self::for_held_factor_accumulator) and
    /// [`for_rescaled_accumulator`](Self::for_rescaled_accumulator):
    /// - held by a plane group (`holder`): the plane group MMA;
    /// - an operand already held in fragments: their instruction, which reads them as they lie;
    /// - both operands codes the device's block-scaled instruction of that shape reads as stored:
    ///   that instruction;
    /// - an operand the TMA engine lands, codes kept beside their scales, which this crate's
    ///   transport decodes in lane runs as it loads, or a sum contracted again where the target
    ///   lays a fragment's `A` like its accumulator: this crate's transport first;
    /// - a sum of products of a factor held in registers: this crate's transport alone;
    /// - a sum whose rows are scaled between its products, where this crate's transport offers the
    ///   call the fragment API would take: this crate's transport, the same call;
    /// - a product of 8-bit floats, whose scales its sum applies cell by cell: this crate's
    ///   transport alone, and `None` where it offers no call;
    /// - otherwise the fragment API first.
    ///
    /// A packed operand decoded on the way contracts at the other operand's element.
    ///
    /// The fragment API and this crate's transport each take the leaf whole where the device
    /// offers that shape, or cut it into calls of their largest shape that does: a leaf one call
    /// fills comes before one cut into several, whichever comes first above, so a leaf the size
    /// of an instruction contracts through it. Each takes the leaf's whole depth where it can,
    /// and only where neither can does the tile walk `K` in the steps of the deepest instruction
    /// that cuts it: a leaf some call takes at its depth keeps that call. The pick is an
    /// instruction at the shape of one call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_plane(
        device: &DeviceProperties,
        target: &TargetProperties,
        holder: ComputeScope,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        [m, n, k]: [usize; 3],
        opened_by: OpenedBy,
    ) -> Option<ShapedInstruction> {
        let whole = |instruction| ShapedInstruction {
            instruction,
            shape: [m, n, k],
        };
        if holder == ComputeScope::MMA_PLANE_GROUP {
            return Some(whole(Instruction::Wgmma));
        }
        if let FragmentSource::Fragments { instruction, cell } = lhs {
            return held_run(device, target, instruction, cell, rhs, sum, [m, n, k]);
        }
        if let FragmentSource::Fragments { instruction, .. } = rhs {
            return Some(whole(instruction));
        }
        if let Some(form) = block_scaled(device, lhs, rhs, [m, n, k]) {
            return Some(whole(Instruction::Mma {
                io: MmaIo::default(),
                operands: MmaOperands::BlockScaled(form),
            }));
        }
        let (lhs_element, rhs_element) = (lhs.element(rhs)?, rhs.element(lhs)?);
        let calls = offered(device, target, lhs_element, rhs_element, sum, [m, n, k])?;
        let read_by_mma = |source| {
            matches!(
                source,
                FragmentSource::Landed(_)
                    | FragmentSource::Codes { .. }
                    | FragmentSource::StoredCodes(_)
            )
        };
        let mma_reads = read_by_mma(lhs) || read_by_mma(rhs);
        // A sum contracted again is kept in the crate's transport only where that transport offers
        // a call the fragment API does not. Where both offer the same calls, as Metal's one
        // `8 × 8 × 8`, the sum is drained like any other: kept in the crate's transport, the next
        // product reads the held cells in an unrolled walk, which triples the compile time of
        // attention's autotune on an M2 (417 s against 1360 s, 1.4 against 2.6 GB) for a speed no
        // one has measured.
        let mma_adds_calls = {
            let matmul = &device.features.matmul;
            let product = |s: &&MmaConfig| {
                s.a_type == lhs_element && s.b_type == rhs_element && s.cd_type == sum
            };
            matmul
                .mma
                .iter()
                .filter(product)
                .any(|shape| !matmul.cmma.iter().filter(product).any(|c| c == shape))
        };
        // An 8-bit float product carries its range in scales outside its values, a row's and a
        // column's, which its sum applies cell by cell as it drains: only this crate's transport
        // holds each unit's cells where it can read them, the fragment API where only it knows.
        let is_8_bit_float = |element: ElemType| {
            matches!(element, ElemType::Float(_)) && element.size_bits() == u8::BITS as usize
        };
        let cells_scaled = is_8_bit_float(lhs_element) && is_8_bit_float(rhs_element);
        let mma_first = mma_reads
            || (opened_by == OpenedBy::StageAccumulator
                && accumulator_is_a(target)
                && mma_adds_calls);
        let mma = Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::Plain,
        };
        // A rescaled sum takes this crate's transport only at the fragment API's own call, so the
        // tile cuts its leaf as it would have: on CUDA, whose manual `16 × 8` cuts a `16 × 16`
        // cell in two, the sum stays in the fragment API.
        let scaled_in_place = |calls: [Option<Call>; 2]| {
            opened_by == OpenedBy::RescaledAccumulator && calls[1].is_some() && calls[0] == calls[1]
        };
        // Each transport's calls in the order it is preferred; within them, one call that fills
        // the leaf first.
        let pick = |calls: [Option<Call>; 2]| {
            let [cmma_call, mma_call] = calls;
            let picks = [
                cmma_call
                    .filter(|_| !cells_scaled && opened_by != OpenedBy::HeldFactorAccumulator)
                    .map(|call| call.picked(Instruction::Cmma)),
                mma_call.map(|call| call.picked(mma)),
            ];
            let ordered = match mma_first || scaled_in_place(calls) {
                true => [picks[1], picks[0]],
                false => picks,
            };
            let one_call = ordered
                .into_iter()
                .flatten()
                .find(|pick| pick.shape == [m, n, k]);
            one_call.or_else(|| ordered.into_iter().flatten().next())
        };
        // A call as deep as the leaf comes before one the tile walks along `K`, whatever the
        // transport: a leaf some call contracts at its depth keeps that call.
        pick(calls.whole).or_else(|| pick(calls.walked))
    }
}

/// An instruction at the `m × n × k` of one of its calls: the whole leaf, or one of the cells the
/// tile cuts the leaf into, its whole depth or a step of the walk the tile takes along it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShapedInstruction {
    pub instruction: Instruction,
    pub shape: [usize; 3],
}

impl ShapedInstruction {
    /// The `m × n` of the output one call fills.
    pub fn cell(&self) -> [usize; 2] {
        let [m, n, _] = self.shape;
        [m, n]
    }

    /// The depth of the contraction one call takes: the leaf's whole depth, or one step of the
    /// walk the tile takes along it.
    pub fn depth(&self) -> usize {
        self.shape[2]
    }
}

/// One transport's call over a leaf: the cell it fills and the depth of the contraction it takes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Call {
    cell: [usize; 2],
    depth: usize,
}

impl Call {
    /// This call through `instruction`.
    fn picked(self, instruction: Instruction) -> ShapedInstruction {
        let [m, n] = self.cell;
        ShapedInstruction {
            instruction,
            shape: [m, n, self.depth],
        }
    }
}

/// The calls the fragment API and the crate's own transport each offer over a leaf, `[cmma, mma]`:
/// those that take its whole depth, and those the tile walks along `K` in steps of their own.
struct Offered {
    whole: [Option<Call>; 2],
    walked: [Option<Call>; 2],
}

/// The calls of a product whose `A` is held in fragments of `held`, each an `m × n` cell of a sum
/// opened before ([`FragmentSource::Fragments`]), over a leaf of `m × n × k` against `rhs`, summed
/// in `sum`: the held instruction, at the device's shapes of it for the held cells' rows and
/// `rhs`'s element, its cell the leaf's own `m × n` where offered, else its largest that cuts the
/// leaf. Each call reads a run of the held cells along `k` as its `A`, so it is as deep as a whole
/// number of them: the leaf's own depth where offered, else the deepest that cuts it, which the
/// tile walks. `None` where the device offers none; where the held cells are the fragment API's,
/// whose cells no call rereads; where a held cell does not lie where an `A` does: on a target
/// laying a fragment's `A` otherwise than its accumulator, or for values wider than 16 bits,
/// whose `A` lays its cells otherwise (`accumulators_as_a`); and for 8-bit values other than
/// `e4m3`. The held cells are cast to the values' element, and `e5m2`'s two bits of mantissa are
/// too coarse for an attention's probabilities. An `A` is a run of them where the target's
/// registers lay the two alike ([`rereads`]). A run of RDNA4's held cells is one cell deep
/// ([`deepest_run`]).
fn held_run(
    device: &DeviceProperties,
    target: &TargetProperties,
    held: Instruction,
    [held_rows, held_width]: [usize; 2],
    rhs: FragmentSource,
    sum: ElemType,
    [m, n, k]: [usize; 3],
) -> Option<ShapedInstruction> {
    let Instruction::Mma { .. } = held else {
        return None;
    };
    let element = rhs.element(rhs)?;
    let holds_probabilities =
        element.size_bits() == 16 || element == ElemType::Float(FloatKind::E4M3);
    if !accumulator_is_a(target) || !holds_probabilities {
        return None;
    }
    let runs: Vec<&MmaConfig> = device
        .features
        .matmul
        .mma
        .iter()
        .filter(|s| s.a_type == element && s.b_type == element && s.cd_type == sum)
        .filter(|s| s.m as usize == held_rows && m.is_multiple_of(held_rows))
        .filter(|s| n.is_multiple_of(s.n as usize))
        .filter(|s| (s.k as usize).is_multiple_of(held_width) && k.is_multiple_of(s.k as usize))
        .filter(|s| s.k as usize <= deepest_run(held_width))
        .filter(|s| rereads(target, [held_rows, held_width], s))
        .collect();
    let depth = runs
        .iter()
        .map(|s| s.k as usize)
        .find(|&depth| depth == k)
        .or_else(|| runs.iter().map(|s| s.k as usize).max())?;
    let columns = runs
        .iter()
        .filter(|s| s.k as usize == depth)
        .map(|s| s.n as usize);
    let width = columns
        .clone()
        .find(|&c| c == n)
        .or_else(|| columns.max())?;
    Some(ShapedInstruction {
        instruction: held,
        shape: [held_rows, width, depth],
    })
}

/// Whether `call`'s `A` is a run of accumulators of `m × n` cells as `target` lays the two in a
/// lane's vectors ([`KeyOrder::rereading`]).
fn rereads(target: &TargetProperties, [m, n]: [usize; 2], call: &MmaConfig) -> bool {
    let element = call.a_type;
    let vectors = |ident, shape: MatrixShape, copies: usize| {
        let width = target.mma.vector_size(ident, shape, element);
        let cells = shape.num_elems(ident) * copies / target.mma.const_plane_lane_count as usize;
        LaneVectors {
            width,
            count: cells / width,
        }
    };
    let acc = vectors(
        MatrixIdent::Accumulator,
        (m, n, n).into(),
        target.mma.register_copy_count_acc,
    );
    let a = vectors(
        MatrixIdent::A,
        (call.m as usize, call.n as usize, call.k as usize).into(),
        target.mma.register_copy_count_a,
    );
    KeyOrder::rereading(element.size(), acc, a).is_some()
}

/// The width of RDNA's WMMA accumulator, `16 × 16`: each half of its lanes holds half of a cell's
/// columns.
const RDNA_WMMA_WIDTH: usize = 16;

/// The deepest run of held cells `held_width` wide one call reads as its `A`. A call deeper than
/// its instruction is cut from each lane's run of `k`, each half of RDNA's lanes holding a half of
/// the whole run; its held cells hold a half of each cell's columns instead, so two of them side
/// by side are not that `A`. There a run is one cell deep, an instruction a cell; elsewhere
/// (`mma.sync`'s `16 × 8`, whose `A` holds its cells' columns in order) any depth.
fn deepest_run(held_width: usize) -> usize {
    match held_width {
        RDNA_WMMA_WIDTH => held_width,
        _ => usize::MAX,
    }
}

/// Whether `target` lays a manual-mma fragment's `A` as it lays its accumulator, so a sum held in
/// registers is the `A` of the next product as it lies.
fn accumulator_is_a(target: &TargetProperties) -> bool {
    let layout = &target.mma;
    layout.register_layout_a == layout.register_layout_acc
        && layout.register_copy_count_a == layout.register_copy_count_acc
}

/// The form of the device's block-scaled instruction of `m × n × k` that reads both operands as
/// stored: codes of the kinds its registers hold, under scales of its type where they are stored
/// as words of them. `None` unless both are such codes.
fn block_scaled(
    device: &DeviceProperties,
    lhs: FragmentSource,
    rhs: FragmentSource,
    [m, n, k]: [usize; 3],
) -> Option<BlockScaledOperands> {
    let (
        FragmentSource::Codes {
            code: lhs_code,
            scales: lhs_scales,
        },
        FragmentSource::Codes {
            code: rhs_code,
            scales: rhs_scales,
        },
    ) = (lhs, rhs)
    else {
        return None;
    };
    let reads =
        |code: QuantValue, scales: Option<Field>, held: FloatKind, form: &BlockScaledOperands| {
            code_of(held) == Some(code)
                && scales.is_none_or(|field| field == Field::of_float(form.scales))
        };
    device
        .features
        .matmul
        .scaled_mma
        .iter()
        .find_map(|instruction| {
            let form = BlockScaledOperands::of(instruction)?;
            let shape = [instruction.m, instruction.n, instruction.k].map(|e| e as usize);
            (shape == [m, n, k]
                && reads(lhs_code, lhs_scales, form.lhs, &form)
                && reads(rhs_code, rhs_scales, form.rhs, &form))
            .then_some(form)
        })
}

/// The code a register element of a block-scaled instruction holds: `e2m1x2` two `e2m1`. `None`
/// for one no stored field holds.
pub(crate) fn code_of(kind: FloatKind) -> Option<QuantValue> {
    match kind {
        FloatKind::E2M1x2 | FloatKind::E2M1 => Some(QuantValue::E2M1),
        FloatKind::E4M3 => Some(QuantValue::E4M3),
        FloatKind::E5M2 => Some(QuantValue::E5M2),
        _ => None,
    }
}

/// The calls the fragment API and the crate's own transport each offer for `lhs · rhs` summed in
/// `sum` over a leaf of `m × n × k`, where `device` offers one ([`Offered`]). A call fills the
/// leaf's own `m × n` where the device offers that shape, else its largest shape whose edges each
/// cut the leaf's (the fewest calls), the squarest of those. It takes the leaf's whole depth
/// where its instruction is `k` deep, or where the crate's transport loads a deeper fragment of
/// 16-bit values in lane runs on `target` and `k` is whole calls of it. Otherwise the deepest
/// instruction whose depth cuts `k` is walked along it in its own steps. `None` where its planes
/// have no single width.
fn offered(
    device: &DeviceProperties,
    target: &TargetProperties,
    lhs: ElemType,
    rhs: ElemType,
    sum: ElemType,
    [m, n, k]: [usize; 3],
) -> Option<Offered> {
    let hardware = &device.hardware;
    let planes_of_one_width = device.features.plane.contains(Plane::Ops)
        && hardware.plane_size_min == hardware.plane_size_max;
    if !planes_of_one_width {
        return None;
    }
    let deep_loads = LaneMap::try_from(target).is_ok() && lhs.size_bits() == 16;
    let elements = |s: &&MmaConfig| s.a_type == lhs && s.b_type == rhs && s.cd_type == sum;
    let cuts = |s: &&MmaConfig| m.is_multiple_of(s.m as usize) && n.is_multiple_of(s.n as usize);
    let walks = |s: &&MmaConfig| k.is_multiple_of(s.k as usize);
    // The deepest instruction cutting `k`, as one step of the tile's walk.
    let deepest = |shapes: &[&MmaConfig]| {
        let depth = shapes.iter().map(|s| s.k as usize).max()?;
        best_call(
            shapes.iter().filter(|s| s.k as usize == depth),
            [m, n],
            depth,
        )
    };
    let offered = &device.features.matmul;
    let cmma: Vec<&MmaConfig> = offered
        .cmma
        .iter()
        .filter(elements)
        .filter(cuts)
        .filter(walks)
        .collect();
    let mma: Vec<&MmaConfig> = offered
        .mma
        .iter()
        .filter(elements)
        .filter(cuts)
        .filter(walks)
        .collect();
    Some(Offered {
        whole: [
            best_call(cmma.iter().filter(|s| s.k as usize == k), [m, n], k),
            best_call(
                mma.iter().filter(|s| s.k as usize == k || deep_loads),
                [m, n],
                k,
            ),
        ],
        walked: [deepest(&cmma), deepest(&mma)],
    })
}

/// The call among `shapes` over a leaf of `m × n`, `depth` deep: the leaf's own `m × n` where one
/// fills it, else the cell that cuts it in the fewest calls, the squarest of those.
fn best_call<'a>(
    shapes: impl Iterator<Item = &'a &'a MmaConfig>,
    [m, n]: [usize; 2],
    depth: usize,
) -> Option<Call> {
    shapes
        .map(|s| [s.m as usize, s.n as usize])
        .max_by_key(|&[cm, cn]| {
            (
                [cm, cn] == [m, n],
                cm * cn,
                core::cmp::Reverse(cm.abs_diff(cn)),
            )
        })
        .map(|cell| Call { cell, depth })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cubecl::cmma::MatrixLayout;
    use cubecl::ir::{
        ContiguousElements, DeviceIdentity, HardwareProperties, MemoryDeviceProperties, VectorSize,
        features::Features,
    };
    use cubecl::profile::TimingMethod;

    pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
    pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A device of 32-unit planes offering no instruction.
    pub(crate) fn plain() -> DeviceProperties {
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 32,
            max_shared_memory_size: 65536,
            max_cube_count: (u32::MAX, u32::from(u16::MAX), u32::from(u16::MAX)),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            num_cpu_physical_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };
        let mut device = DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(u64::MAX, 256),
            hardware,
            TimingMethod::Device,
            DeviceIdentity {
                name: "test".to_string(),
                fingerprint: "test".to_string(),
                physical: None,
            },
        );
        device.features.plane.insert(Plane::Ops);
        device
    }

    pub(crate) fn shape(m: u32, n: u32, k: u32) -> MmaConfig {
        MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m,
            n,
            k,
        }
    }

    /// An M2-like device: `simdgroup_matrix` through both transports, `8 × 8 × 8`.
    pub(crate) fn metal() -> DeviceProperties {
        let mut device = plain();
        device.features.matmul.cmma.insert(shape(8, 8, 8));
        device.features.matmul.mma.insert(shape(8, 8, 8));
        device
    }

    /// An sm_80-like device: `wmma`'s three `f16` shapes and the manual `16 × 8 × 16`.
    pub(crate) fn cuda() -> DeviceProperties {
        let mut device = plain();
        for (m, n, k) in [(16, 16, 16), (32, 8, 16), (8, 32, 16)] {
            device.features.matmul.cmma.insert(shape(m, n, k));
        }
        device.features.matmul.mma.insert(shape(16, 8, 16));
        device
    }

    /// An RDNA-like device: `wmma` and the manual mma both at `16 × 16 × 16`.
    pub(crate) fn rdna() -> DeviceProperties {
        let mut device = plain();
        device.features.matmul.cmma.insert(shape(16, 16, 16));
        device.features.matmul.mma.insert(shape(16, 16, 16));
        device
    }

    /// A target whose `A` and accumulator lie alike, as on CUDA and Metal.
    pub(crate) fn sum_is_next_a() -> TargetProperties {
        let mut target = TargetProperties::default();
        target.mma.register_layout_a = target.mma.register_layout_acc;
        target.mma.register_copy_count_a = target.mma.register_copy_count_acc;
        target
    }

    /// A target whose `A` lies otherwise than its accumulator, as on RDNA3.
    pub(crate) fn sum_is_not_next_a() -> TargetProperties {
        let mut target = sum_is_next_a();
        target.mma.register_layout_a = match target.mma.register_layout_acc {
            MatrixLayout::RowMajor => MatrixLayout::ColMajor,
            _ => MatrixLayout::RowMajor,
        };
        target
    }

    /// A target whose `A` and accumulator lie alike, a lane's vectors of each as long as given.
    fn sum_is_next_a_in(a: VectorSize, acc: VectorSize) -> TargetProperties {
        let mut target = sum_is_next_a();
        target.mma.contiguous_elements = ContiguousElements::new(move |_, ident, _| match ident {
            MatrixIdent::A | MatrixIdent::B => a,
            MatrixIdent::Accumulator => acc,
        });
        target
    }

    pub(crate) fn mma() -> Instruction {
        Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::Plain,
        }
    }

    pub(crate) const HALF: FragmentSource = FragmentSource::Values(F16);

    fn picked(
        device: &DeviceProperties,
        lhs: FragmentSource,
        shape: [usize; 3],
    ) -> Option<Instruction> {
        Instruction::for_accumulator(
            device,
            &sum_is_next_a(),
            ComputeScope::Plane,
            lhs,
            HALF,
            F32,
            shape,
        )
        .map(|call| call.instruction)
    }

    /// A drained sum goes through the vendor's fragment API.
    #[test]
    fn an_accumulators_sum_goes_through_cmma() {
        assert_eq!(picked(&metal(), HALF, [8, 8, 8]), Some(Instruction::Cmma));
        assert_eq!(picked(&cuda(), HALF, [16, 16, 16]), Some(Instruction::Cmma));
    }

    /// What a rescaled sum over `shape` contracts through on `device`.
    fn rescaled(device: &DeviceProperties, shape: [usize; 3]) -> Option<Instruction> {
        Instruction::for_rescaled_accumulator(device, &sum_is_not_next_a(), HALF, HALF, F32, shape)
            .map(|call| call.instruction)
    }

    /// A sum scaled along its rows is held where its units read the rows, wherever the manual mma
    /// offers the fragment API's call.
    #[test]
    fn a_rescaled_sum_takes_the_manual_mma_at_the_fragment_apis_call() {
        assert_eq!(rescaled(&rdna(), [16, 16, 16]), Some(mma()));
        assert_eq!(rescaled(&metal(), [8, 8, 8]), Some(mma()));
    }

    /// Where the manual mma would cut the fragment API's cell otherwise, as CUDA's `16 × 8` cuts a
    /// `16 × 16`, a rescaled sum stays in the fragment API.
    #[test]
    fn a_rescaled_sum_stays_in_cmma_where_the_manual_mma_cuts_another_call() {
        assert_eq!(rescaled(&cuda(), [16, 16, 16]), Some(Instruction::Cmma));
    }

    const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
    const FP8: FragmentSource = FragmentSource::Values(E4M3);

    /// An RDNA4-like device offering `e4m3` products at `16 × 16 × 16` through the manual mma, and
    /// through the fragment API where `cmma` says.
    fn fp8_device(cmma: bool) -> DeviceProperties {
        let mut device = plain();
        let fp8 = MmaConfig {
            a_type: E4M3,
            b_type: E4M3,
            ..shape(16, 16, 16)
        };
        device.features.matmul.mma.insert(fp8);
        if cmma {
            device.features.matmul.cmma.insert(fp8);
        }
        device
    }

    /// What a drained `e4m3` sum over `shape` contracts through on `device`.
    fn fp8_sum(device: &DeviceProperties, shape: [usize; 3]) -> Option<Instruction> {
        Instruction::for_accumulator(
            device,
            &sum_is_next_a(),
            ComputeScope::Plane,
            FP8,
            FP8,
            F32,
            shape,
        )
        .map(|call| call.instruction)
    }

    /// An `e4m3` sum, whose scales its drain applies cell by cell, is held in the manual mma, even
    /// where the fragment API's call fills the leaf as well.
    #[test]
    fn an_8_bit_float_sum_takes_the_manual_mma_over_a_whole_leaf_cmma_call() {
        assert_eq!(fp8_sum(&fp8_device(true), [16, 16, 16]), Some(mma()));
    }

    /// Where only the fragment API multiplies `e4m3`, an `e4m3` sum has no instruction: its cells
    /// would lie where the drain cannot scale them.
    #[test]
    fn an_8_bit_float_sum_has_no_instruction_without_the_manual_mma() {
        let mut device = plain();
        device.features.matmul.cmma.insert(MmaConfig {
            a_type: E4M3,
            b_type: E4M3,
            ..shape(16, 16, 16)
        });
        assert_eq!(fp8_sum(&device, [16, 16, 16]), None);
    }

    /// What a drained sum over `shape` contracts through, at the shape of one call.
    fn cut(
        device: &DeviceProperties,
        lhs: FragmentSource,
        shape: [usize; 3],
    ) -> Option<ShapedInstruction> {
        Instruction::for_plane(
            device,
            &sum_is_next_a(),
            ComputeScope::Plane,
            lhs,
            HALF,
            F32,
            shape,
            OpenedBy::Accumulator,
        )
    }

    /// A leaf no instruction fills in one call is cut into calls of the largest shape that cuts
    /// it, the squarest of those: the plane's tile is the kernel's, the cells the tile's.
    #[test]
    fn a_leaf_wider_than_any_instruction_is_cut_into_its_largest_cells() {
        let (metal, cuda) = (metal(), cuda());
        let pick =
            |device, shape| cut(device, HALF, shape).map(|call| (call.instruction, call.cell()));
        assert_eq!(pick(&metal, [16, 32, 8]), Some((Instruction::Cmma, [8, 8])));
        assert_eq!(
            pick(&cuda, [32, 32, 16]),
            Some((Instruction::Cmma, [16, 16]))
        );
        // `32 × 8` fills a `32 × 8` leaf in one call, ahead of `16 × 8` cutting it in two.
        assert_eq!(pick(&cuda, [32, 8, 16]), Some((Instruction::Cmma, [32, 8])));
        // Only the manual mma cuts `16 × 24`, three calls along `n`.
        assert_eq!(pick(&cuda, [16, 24, 16]), Some((mma(), [16, 8])));
        // No instruction is 8 deep in `f16` there.
        assert_eq!(pick(&cuda, [32, 8, 8]), None);
    }

    /// A leaf deeper than every instruction is walked along `K` in the steps of the deepest one
    /// that cuts its depth, at that one's best cell: the plane's depth is the kernel's, the steps
    /// the tile's. Devices offering only the fragment API, which loads no fragment deeper than its
    /// instruction.
    #[test]
    fn a_leaf_deeper_than_any_instruction_is_walked_in_its_steps() {
        let fragment_api_only = |mut device: DeviceProperties| {
            device.features.matmul.mma.clear();
            device
        };
        let (metal, cuda) = (fragment_api_only(metal()), fragment_api_only(cuda()));
        let pick =
            |device, shape| cut(device, HALF, shape).map(|p| (p.instruction, p.cell(), p.depth()));
        assert_eq!(
            pick(&metal, [16, 32, 32]),
            Some((Instruction::Cmma, [8, 8], 8))
        );
        assert_eq!(
            pick(&cuda, [32, 32, 64]),
            Some((Instruction::Cmma, [16, 16], 16))
        );
        // Not a whole number of steps of any instruction.
        assert_eq!(pick(&cuda, [16, 16, 24]), None);
    }

    /// A call that takes the leaf's whole depth comes before one the tile walks, whichever the rule
    /// would otherwise put first: where the crate's transport loads 16-bit fragments deeper than
    /// its instruction, as on the mocked target, the manual mma takes a deep leaf whole, ahead of
    /// the fragment API walked.
    #[test]
    fn a_call_taking_the_whole_depth_comes_before_a_walked_one() {
        let pick = Instruction::for_accumulator(
            &cuda(),
            &sum_is_next_a(),
            ComputeScope::Plane,
            HALF,
            HALF,
            F32,
            [16, 16, 64],
        );
        assert_eq!(
            pick,
            Some(ShapedInstruction {
                instruction: mma(),
                shape: [16, 8, 64],
            })
        );
    }

    /// An instruction of the leaf's own shape fills it in one call, ahead of one the rule would
    /// otherwise put first but which only cuts it: every leaf the size of an instruction keeps
    /// its instruction.
    #[test]
    fn a_leaf_one_call_fills_takes_that_instruction() {
        let landed = FragmentSource::Landed(F16);
        let pick = |shape| cut(&cuda(), landed, shape).map(|call| (call.instruction, call.cell()));
        assert_eq!(pick([16, 8, 16]), Some((mma(), [16, 8])));
        assert_eq!(pick([16, 16, 16]), Some((Instruction::Cmma, [16, 16])));
        assert_eq!(pick([32, 16, 16]), Some((mma(), [16, 8])));
    }

    /// A sum contracted again goes through the crate's transport where it is the next `A` and the
    /// transport offers a call the fragment API does not, as `sm_80`'s `16 × 8 × 16`; where both
    /// offer the same calls, as Metal's `8 × 8 × 8`, it is drained like any sum, through the
    /// fragment API.
    #[test]
    fn a_stage_accumulators_sum_goes_through_mma_where_mma_adds_a_call() {
        let pick = |device, shape| {
            Instruction::for_stage_accumulator(&device, &sum_is_next_a(), HALF, HALF, F32, shape)
                .map(|pick| pick.instruction)
        };
        assert_eq!(pick(cuda(), [16, 8, 16]), Some(mma()));
        assert_eq!(pick(metal(), [8, 8, 8]), Some(Instruction::Cmma));
    }

    /// A sum of a factor held in registers goes through the crate's transport where the fragment
    /// API offers the same calls too, and nowhere else.
    #[test]
    fn a_held_factors_sum_goes_through_mma_alone() {
        let pick = |device: &DeviceProperties, shape| {
            Instruction::for_held_factor_accumulator(
                device,
                &sum_is_next_a(),
                HALF,
                HALF,
                F32,
                shape,
            )
            .map(|pick| pick.instruction)
        };
        assert_eq!(pick(&metal(), [8, 8, 8]), Some(mma()));
        let mut fragment_api_alone = plain();
        fragment_api_alone
            .features
            .matmul
            .cmma
            .insert(shape(16, 16, 16));
        assert_eq!(pick(&fragment_api_alone, [16, 16, 16]), None);
    }

    /// Where the accumulator is not the next `A`, keeping the sum in registers buys nothing.
    #[test]
    fn a_stage_accumulators_sum_needs_to_be_the_next_a() {
        let instruction = Instruction::for_stage_accumulator(
            &cuda(),
            &sum_is_not_next_a(),
            HALF,
            HALF,
            F32,
            [16, 16, 16],
        );
        assert_eq!(
            instruction.map(|pick| pick.instruction),
            Some(Instruction::Cmma)
        );
    }

    /// An operand the TMA engine lands goes through the crate's transport even for a drained
    /// sum; a packed one decoded on the way takes the rule of the element it decodes into.
    #[test]
    fn a_landed_operand_goes_through_mma() {
        let landed = FragmentSource::Landed(F16);
        assert_eq!(picked(&cuda(), landed, [16, 8, 16]), Some(mma()));
        assert_eq!(picked(&metal(), landed, [8, 8, 8]), Some(mma()));
        let decoded = FragmentSource::Decoded;
        assert_eq!(
            picked(&metal(), decoded, [8, 8, 8]),
            Some(Instruction::Cmma)
        );
    }

    /// Codes kept beside their scales against plain values, NVFP4 weights against `f16`
    /// activations, are decoded as each fragment loads: the crate's transport, at the values'
    /// element, where no block-scaled instruction reads both.
    #[test]
    fn codes_against_values_go_through_mma() {
        let codes = FragmentSource::Codes {
            code: QuantValue::E2M1,
            scales: None,
        };
        let pick = Instruction::for_accumulator(
            &metal(),
            &sum_is_next_a(),
            ComputeScope::Plane,
            HALF,
            codes,
            F32,
            [8, 8, 8],
        );
        assert_eq!(pick.map(|call| call.instruction), Some(mma()));
    }

    /// An operand already held in fragments is read as it lies: the sum keeps their instruction.
    #[test]
    fn an_operand_held_in_fragments_keeps_their_instruction() {
        let held = FragmentSource::Fragments {
            instruction: mma(),
            cell: [8, 8],
        };
        assert_eq!(picked(&metal(), held, [8, 8, 8]), Some(mma()));
    }

    /// A product on held cells reads a run of them as each call's `A`: at the held cells' rows,
    /// its leaf's own depth where the device offers it, else the deepest run that cuts the leaf,
    /// which the tile walks. A held `16 × 8` cell on `sm_80` is read two at a time by
    /// `16 × 8 × 16`.
    #[test]
    fn a_product_on_held_cells_reads_runs_of_them() {
        let held = |cell| FragmentSource::Fragments {
            instruction: mma(),
            cell,
        };
        let pick = |device, cell, shape| {
            cut(device, held(cell), shape).map(|p| (p.instruction, p.cell(), p.depth()))
        };
        let (metal, cuda) = (metal(), cuda());
        assert_eq!(pick(&metal, [8, 8], [8, 16, 32]), Some((mma(), [8, 8], 8)));
        assert_eq!(
            pick(&cuda, [16, 8], [16, 32, 16]),
            Some((mma(), [16, 8], 16))
        );
        assert_eq!(
            pick(&cuda, [16, 8], [32, 8, 64]),
            Some((mma(), [16, 8], 16))
        );
        // No call rereads held cells where a fragment's `A` lies otherwise than its accumulator,
        // nor `f32` cells, nor where the device offers no run of them.
        let elsewhere = Instruction::for_accumulator(
            &metal,
            &sum_is_not_next_a(),
            ComputeScope::Plane,
            held([8, 8]),
            HALF,
            F32,
            [8, 8, 8],
        );
        assert_eq!(elsewhere, None);
        let single = FragmentSource::Values(F32);
        assert_eq!(
            Instruction::for_accumulator(
                &metal,
                &sum_is_next_a(),
                ComputeScope::Plane,
                held([8, 8]),
                single,
                F32,
                [8, 8, 8],
            ),
            None
        );
        assert_eq!(pick(&cuda, [16, 8], [16, 8, 24]), None);
    }

    /// Held `f16` cells contract with `e4m3` values on the device's FP8 instruction, four
    /// `16 × 8` cells a `16 × 8 × 32` call, as an FP8 cache's probabilities are cast to its
    /// values' element. `e5m2` values are refused: too coarse for the probabilities.
    #[test]
    fn held_cells_meet_e4m3_values_on_the_fp8_instruction() {
        let fp8 = |kind| ElemType::Float(kind);
        let mut device = cuda();
        for kind in [FloatKind::E4M3, FloatKind::E5M2] {
            device.features.matmul.mma.insert(MmaConfig {
                a_type: fp8(kind),
                b_type: fp8(kind),
                ..shape(16, 8, 32)
            });
        }
        let held = FragmentSource::Fragments {
            instruction: mma(),
            cell: [16, 8],
        };
        let pick = |kind| {
            Instruction::for_accumulator(
                &device,
                &sum_is_next_a_in(4, 2),
                ComputeScope::Plane,
                held,
                FragmentSource::Values(fp8(kind)),
                F32,
                [16, 32, 64],
            )
            .map(|p| (p.instruction, p.cell(), p.depth()))
        };
        assert_eq!(pick(FloatKind::E4M3), Some((mma(), [16, 8], 32)));
        assert_eq!(pick(FloatKind::E5M2), None);
    }

    /// Held `16 × 16` cells, as RDNA's, are reread one cell a call even where the device offers a
    /// deeper call: a run of two would pair the halves of two cells.
    #[test]
    fn held_cells_16_wide_are_reread_a_cell_a_call() {
        let mut device = rdna();
        device.features.matmul.mma.insert(shape(16, 16, 32));
        let held = FragmentSource::Fragments {
            instruction: mma(),
            cell: [16, 16],
        };

        let pick = Instruction::for_accumulator(
            &device,
            &sum_is_next_a(),
            ComputeScope::Plane,
            held,
            HALF,
            F32,
            [16, 64, 32],
        )
        .map(|p| p.depth());

        assert_eq!(pick, Some(16));
    }

    /// Held `16 × 16` cells are reread as an `e4m3` `A` where a lane's vectors of the two are as
    /// long, as RDNA4's are, and not where an `A` vector is four accumulator vectors long.
    #[test]
    fn held_cells_meet_e4m3_values_as_the_registers_lay_them() {
        let mut device = rdna();
        let fp8 = ElemType::Float(FloatKind::E4M3);
        device.features.matmul.mma.insert(MmaConfig {
            a_type: fp8,
            b_type: fp8,
            ..shape(16, 16, 16)
        });
        let held = FragmentSource::Fragments {
            instruction: mma(),
            cell: [16, 16],
        };
        let pick = |target| {
            Instruction::for_accumulator(
                &device,
                &target,
                ComputeScope::Plane,
                held,
                FragmentSource::Values(fp8),
                F32,
                [16, 32, 64],
            )
            .map(|p| (p.instruction, p.cell(), p.depth()))
        };

        assert_eq!(pick(sum_is_next_a_in(8, 8)), Some((mma(), [16, 16], 16)));
        assert_eq!(pick(sum_is_next_a_in(8, 2)), None);
    }

    /// Plane groups of four issue the plane group MMA.
    #[test]
    fn a_plane_group_issues_the_plane_group_mma() {
        let instruction = Instruction::for_accumulator(
            &cuda(),
            &sum_is_next_a(),
            ComputeScope::MMA_PLANE_GROUP,
            HALF,
            HALF,
            F32,
            [64, 128, 16],
        );
        assert_eq!(
            instruction.map(|call| call.instruction),
            Some(Instruction::Wgmma)
        );
    }

    /// A device offering only one transport is reached through it, whatever becomes of the sum.
    #[test]
    fn one_transport_serves_every_sum() {
        let mut device = plain();
        device.features.matmul.mma.insert(shape(16, 8, 16));
        assert_eq!(picked(&device, HALF, [16, 8, 16]), Some(mma()));
        assert_eq!(
            Instruction::for_stage_accumulator(
                &device,
                &sum_is_next_a(),
                HALF,
                HALF,
                F32,
                [16, 8, 16]
            )
            .map(|pick| pick.instruction),
            Some(mma())
        );
    }

    /// No instruction for the elements, planes of no single width, or two operands that name no
    /// element: none.
    #[test]
    fn no_instruction_where_none_fits() {
        let single = FragmentSource::Values(F32);
        let f32_only = Instruction::for_accumulator(
            &cuda(),
            &sum_is_next_a(),
            ComputeScope::Plane,
            single,
            single,
            F32,
            [16, 16, 16],
        );
        assert_eq!(f32_only, None);

        let mut uneven = cuda();
        uneven.hardware.plane_size_min = 16;
        assert_eq!(picked(&uneven, HALF, [16, 16, 16]), None);

        let decoded = FragmentSource::Decoded;
        let none_named = Instruction::for_accumulator(
            &cuda(),
            &sum_is_next_a(),
            ComputeScope::Plane,
            decoded,
            decoded,
            F32,
            [16, 16, 16],
        );
        assert_eq!(none_named, None);
    }
}
