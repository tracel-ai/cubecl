//! What one plane contracts through ([`Instruction`]).

use cubecl::cmma::MatrixIdent;
use cubecl::ir::features::{MmaConfig, Plane};
use cubecl::ir::{
    DeviceProperties, ElemType, FloatKind, TargetProperties, features::ScaledMmaConfig,
};

use cubecl::quant::scheme::QuantValue;

use crate::tile::kind::plane::mma::lane_runs::LaneMap;
use crate::tile::packing::base::Field;
use crate::{ComputeScope, MmaIo, SoftwareMatmul};

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
    /// A product already held in fragments of this instruction, contracted again as it lies.
    Fragments(Instruction),
}

impl FragmentSource {
    /// The element fragments of this operand contract at, against `other`: its own, or the other
    /// operand's for packed values decoded on the way or codes decoded as they load. `None` where
    /// neither operand names one.
    fn element(self, other: FragmentSource) -> Option<ElemType> {
        let named = |source: FragmentSource| match source {
            FragmentSource::Values(element) | FragmentSource::Landed(element) => Some(element),
            FragmentSource::Codes { .. }
            | FragmentSource::StoredCodes(_)
            | FragmentSource::Decoded
            | FragmentSource::Fragments(_) => None,
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
}

impl Instruction {
    /// What the sum [`Tile::accumulator`](crate::Tile::accumulator) opens contracts through on
    /// `device` compiling for `target`, and the cell of its leaf each call fills: a sum in `sum`
    /// held by `holder`, of fragments loaded from `lhs` and `rhs`, over a leaf of `m × n × k`. The
    /// rule the kernel reads as it expands; a host reads it to size what the kernel will hold. `None` where the device offers
    /// no matrix instruction for the product, which a plane routine refuses.
    pub fn for_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        holder: ComputeScope,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<PlanePick> {
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
    /// `A` does.
    pub fn for_stage_accumulator(
        device: &DeviceProperties,
        target: &TargetProperties,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
        shape: [usize; 3],
    ) -> Option<PlanePick> {
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

    /// The rule behind [`for_accumulator`](Self::for_accumulator) and
    /// [`for_stage_accumulator`](Self::for_stage_accumulator):
    /// - held by a plane group (`holder`): the plane group MMA;
    /// - an operand already held in fragments: their instruction, which reads them as they lie;
    /// - both operands codes the device's block-scaled instruction of that shape reads as stored:
    ///   that instruction;
    /// - an operand the TMA engine lands, codes kept beside their scales, which this crate's
    ///   transport decodes in lane runs as it loads, or a sum contracted again where the target
    ///   lays a fragment's `A` like its accumulator: this crate's transport first;
    /// - otherwise the fragment API first.
    ///
    /// A packed operand decoded on the way contracts at the other operand's element.
    ///
    /// The fragment API and this crate's transport each take the leaf whole where the device
    /// offers that shape, or cut it into calls of their largest shape that does: a leaf one call
    /// fills comes before one cut into several, whichever comes first above, so a leaf the size
    /// of an instruction contracts through it. The pick says the cell one call fills.
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
    ) -> Option<PlanePick> {
        let whole = |instruction| PlanePick {
            instruction,
            cell: [m, n],
        };
        if holder == ComputeScope::PLANE_GROUP {
            return Some(whole(Instruction::Wgmma));
        }
        if let FragmentSource::Fragments(held) = lhs {
            return Some(whole(held));
        }
        if let FragmentSource::Fragments(held) = rhs {
            return Some(whole(held));
        }
        if let Some(form) = block_scaled(device, lhs, rhs, [m, n, k]) {
            return Some(whole(Instruction::Mma {
                io: MmaIo::default(),
                operands: MmaOperands::BlockScaled(form),
            }));
        }
        let (lhs_element, rhs_element) = (lhs.element(rhs)?, rhs.element(lhs)?);
        let [cmma, mma] = offered(device, target, lhs_element, rhs_element, sum, [m, n, k])?;
        let read_by_mma = |source| {
            matches!(
                source,
                FragmentSource::Landed(_)
                    | FragmentSource::Codes { .. }
                    | FragmentSource::StoredCodes(_)
            )
        };
        let mma_reads = read_by_mma(lhs) || read_by_mma(rhs);
        let layout = &target.mma;
        let sum_is_next_a = layout.register_layout_a == layout.register_layout_acc
            && layout.register_duplication_a == layout.register_duplication_acc;
        let mma_first = mma_reads || (opened_by == OpenedBy::StageAccumulator && sum_is_next_a);
        let cmma = cmma.map(|cell| PlanePick {
            instruction: Instruction::Cmma,
            cell,
        });
        let mma = mma.map(|cell| PlanePick {
            instruction: Instruction::Mma {
                io: MmaIo::default(),
                operands: MmaOperands::Plain,
            },
            cell,
        });
        let ordered = match mma_first {
            true => [mma, cmma],
            false => [cmma, mma],
        };
        let one_call = ordered
            .into_iter()
            .flatten()
            .find(|pick| pick.cell == [m, n]);
        one_call.or_else(|| ordered.into_iter().flatten().next())
    }
}

/// What a plane's sum contracts through, and the `m × n` cell of its output one call fills: the
/// whole leaf, or one of the cells the tile cuts the leaf into.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PlanePick {
    pub instruction: Instruction,
    pub cell: [usize; 2],
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

/// The cell the fragment API and the crate's own transport each fill per call of `lhs · rhs`
/// summed in `sum` over a leaf of `m × n × k`, where `device` offers one: the leaf's own `m × n`
/// where it offers that shape, else its largest shape whose edges each cut the leaf's (the fewest
/// calls), the squarest of those. The fragment API's depth is `k`; the crate's transport's is `k`,
/// or a depth `k` is whole calls of where it loads a deeper fragment of 16-bit values in lane runs
/// on `target`. `None` where its planes have no single width.
fn offered(
    device: &DeviceProperties,
    target: &TargetProperties,
    lhs: ElemType,
    rhs: ElemType,
    sum: ElemType,
    [m, n, k]: [usize; 3],
) -> Option<[Option<[usize; 2]>; 2]> {
    let hardware = &device.hardware;
    let planes_of_one_width = device.features.plane.contains(Plane::Ops)
        && hardware.plane_size_min == hardware.plane_size_max;
    if !planes_of_one_width {
        return None;
    }
    let deep_loads = LaneMap::try_from(target).is_ok() && lhs.size_bits() == 16;
    let elements = |s: &&MmaConfig| s.a_type == lhs && s.b_type == rhs && s.cd_type == sum;
    let cell = |s: &MmaConfig| [s.m as usize, s.n as usize];
    let cuts = |[cm, cn]: [usize; 2]| m.is_multiple_of(cm) && n.is_multiple_of(cn);
    // The leaf's own shape, else the cutting cell with the fewest calls, the squarest of those.
    let best = |cells: Vec<[usize; 2]>| {
        cells.iter().copied().find(|&c| c == [m, n]).or_else(|| {
            cells
                .iter()
                .copied()
                .max_by_key(|&[cm, cn]| (cm * cn, core::cmp::Reverse(cm.abs_diff(cn))))
        })
    };
    let offered = &device.features.matmul;
    let cmma = offered
        .cmma
        .iter()
        .filter(elements)
        .filter(|s| s.k as usize == k)
        .map(cell)
        .filter(|&c| cuts(c))
        .collect();
    let mma = offered
        .mma
        .iter()
        .filter(elements)
        .filter(|s| {
            let depth = s.k as usize;
            k == depth || (deep_loads && k.is_multiple_of(depth))
        })
        .map(cell)
        .filter(|&c| cuts(c))
        .collect();
    Some([best(cmma), best(mma)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cmma::MatrixLayout;
    use cubecl::ir::{
        DeviceIdentity, HardwareProperties, MemoryDeviceProperties, VectorSize, features::Features,
    };
    use cubecl::profile::TimingMethod;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A device of 32-unit planes offering no instruction.
    fn plain() -> DeviceProperties {
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

    fn shape(m: u32, n: u32, k: u32) -> MmaConfig {
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
    fn metal() -> DeviceProperties {
        let mut device = plain();
        device.features.matmul.cmma.insert(shape(8, 8, 8));
        device.features.matmul.mma.insert(shape(8, 8, 8));
        device
    }

    /// An sm_80-like device: `wmma`'s three `f16` shapes and the manual `16 × 8 × 16`.
    fn cuda() -> DeviceProperties {
        let mut device = plain();
        for (m, n, k) in [(16, 16, 16), (32, 8, 16), (8, 32, 16)] {
            device.features.matmul.cmma.insert(shape(m, n, k));
        }
        device.features.matmul.mma.insert(shape(16, 8, 16));
        device
    }

    /// A target whose `A` and accumulator lie alike, as on CUDA and Metal.
    fn sum_is_next_a() -> TargetProperties {
        let mut target = TargetProperties::default();
        target.mma.register_layout_a = target.mma.register_layout_acc;
        target.mma.register_duplication_a = target.mma.register_duplication_acc;
        target
    }

    /// A target whose `A` lies otherwise than its accumulator, as on RDNA.
    fn sum_is_not_next_a() -> TargetProperties {
        let mut target = sum_is_next_a();
        target.mma.register_layout_a = match target.mma.register_layout_acc {
            MatrixLayout::RowMajor => MatrixLayout::ColMajor,
            _ => MatrixLayout::RowMajor,
        };
        target
    }

    fn mma() -> Instruction {
        Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::Plain,
        }
    }

    const HALF: FragmentSource = FragmentSource::Values(F16);

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
        .map(|pick| pick.instruction)
    }

    /// A drained sum goes through the vendor's fragment API.
    #[test]
    fn an_accumulators_sum_goes_through_cmma() {
        assert_eq!(picked(&metal(), HALF, [8, 8, 8]), Some(Instruction::Cmma));
        assert_eq!(picked(&cuda(), HALF, [16, 16, 16]), Some(Instruction::Cmma));
    }

    /// The pick for a drained sum over `shape`, its cell included.
    fn cut(device: &DeviceProperties, lhs: FragmentSource, shape: [usize; 3]) -> Option<PlanePick> {
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
        let pick = |device, shape| cut(device, HALF, shape).map(|p| (p.instruction, p.cell));
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

    /// An instruction of the leaf's own shape fills it in one call, ahead of one the rule would
    /// otherwise put first but which only cuts it: every leaf the size of an instruction keeps
    /// its instruction.
    #[test]
    fn a_leaf_one_call_fills_takes_that_instruction() {
        let landed = FragmentSource::Landed(F16);
        let pick = |shape| cut(&cuda(), landed, shape).map(|p| (p.instruction, p.cell));
        assert_eq!(pick([16, 8, 16]), Some((mma(), [16, 8])));
        assert_eq!(pick([16, 16, 16]), Some((Instruction::Cmma, [16, 16])));
        assert_eq!(pick([32, 16, 16]), Some((mma(), [16, 8])));
    }

    /// A sum contracted again goes through the crate's transport where it is the next `A`.
    #[test]
    fn a_stage_accumulators_sum_goes_through_mma() {
        for (device, shape) in [(metal(), [8, 8, 8]), (cuda(), [16, 8, 16])] {
            let instruction = Instruction::for_stage_accumulator(
                &device,
                &sum_is_next_a(),
                HALF,
                HALF,
                F32,
                shape,
            );
            assert_eq!(instruction.map(|p| p.instruction), Some(mma()));
        }
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
        assert_eq!(instruction.map(|p| p.instruction), Some(Instruction::Cmma));
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
        assert_eq!(pick.map(|p| p.instruction), Some(mma()));
    }

    /// An operand already held in fragments is read as it lies: the sum keeps their instruction.
    #[test]
    fn an_operand_held_in_fragments_keeps_their_instruction() {
        let held = FragmentSource::Fragments(mma());
        assert_eq!(picked(&metal(), held, [8, 8, 8]), Some(mma()));
    }

    /// Plane groups of four issue the plane group MMA.
    #[test]
    fn a_plane_group_issues_the_plane_group_mma() {
        let instruction = Instruction::for_accumulator(
            &cuda(),
            &sum_is_next_a(),
            ComputeScope::PLANE_GROUP,
            HALF,
            HALF,
            F32,
            [64, 128, 16],
        );
        assert_eq!(instruction.map(|p| p.instruction), Some(Instruction::Wgmma));
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
            .map(|p| p.instruction),
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
