//! What a host reads of a product's Step ([`ProductStep`]): the Steps a device allows and what
//! computing one spends ([`StepCost`]). The Lowering picks what computes a Step; a host never
//! sees it.

use core::cmp::Reverse;

use cubecl::ir::features::MmaConfig;
use cubecl::ir::{DeviceProperties, ElemType, FloatKind, TargetProperties};

use super::instruction::{FragmentSource, Instruction, OpenedBy, ShapedInstruction};
use crate::launch::Delivery;
use crate::{ComputeScope, Scratch, StageStorage};

/// The Step of a product a Scope holds: `m × n × k` of fragments loaded from `lhs` and `rhs`,
/// summed in `sum`. A kernel's innermost loop covers it whole, and the Lowering computes it with
/// what the device offers, one call or several.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProductStep {
    pub shape: [usize; 3],
    pub holder: ComputeScope,
    pub lhs: FragmentSource,
    pub rhs: FragmentSource,
    pub sum: ElemType,
}

/// What computing a [`ProductStep`] spends that a host sizes a launch by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StepCost {
    /// The cell a drained sum is spilled through shared memory in, one call's output; `None`
    /// where each unit holds its own cells and writes them where they go.
    pub bounced_cell: Option<[usize; 2]>,
}

impl ProductStep {
    /// The Steps `device` allows a plane for a product of `lhs` by `rhs` summed in `sum`: the
    /// shapes of its matrix instructions for those elements, the one that contracts the most
    /// first, then the squarest. One on an M2 (`8 × 8 × 8`) and on RDNA (`16 × 16 × 16`), several
    /// on NVIDIA, among which autotune measures. None for a plane group, whose Step is its own
    /// instruction's, nor where either operand names no element.
    pub fn sizes(
        device: &DeviceProperties,
        holder: ComputeScope,
        lhs: FragmentSource,
        rhs: FragmentSource,
        sum: ElemType,
    ) -> Vec<[usize; 3]> {
        if holder != ComputeScope::Plane {
            return Vec::new();
        }
        let (Some(lhs), Some(rhs)) = (lhs.element(rhs), rhs.element(lhs)) else {
            return Vec::new();
        };
        let matmul = &device.features.matmul;
        let mut sizes: Vec<[usize; 3]> = matmul
            .cmma
            .iter()
            .chain(matmul.mma.iter())
            .filter(|config| products(config, lhs, rhs, sum))
            .map(|config| [config.m, config.n, config.k].map(|edge| edge as usize))
            .collect();
        sizes.sort_by_key(|&[m, n, k]| (Reverse(m * n * k), m.abs_diff(n), m, n, k));
        sizes.dedup();
        sizes
    }

    /// What computing this Step spends where its sum is drained into an output
    /// ([`Tile::accumulator`](crate::Tile::accumulator)). `None` where the Lowering cannot compute
    /// it on `device` compiling for `target`, which a host Declines.
    pub fn cost(&self, device: &DeviceProperties, target: &TargetProperties) -> Option<StepCost> {
        self.call(device, target, OpenedBy::Accumulator)
            .map(|call| StepCost::of(call.instruction, call.cell()))
    }

    /// What computing `next` spends where it reads this Step's sum as its `A` where the sum lies,
    /// in the registers of the units that hold it, as attention's value reads its score
    /// ([`Tile::stage_accumulator`](crate::Tile::stage_accumulator)). `None` where the sum cannot
    /// stay there for `next`, which then reads it through shared memory.
    pub fn kept_for(
        &self,
        next: &ProductStep,
        device: &DeviceProperties,
        target: &TargetProperties,
    ) -> Option<StepCost> {
        let kept = self.call(device, target, OpenedBy::StageAccumulator)?;
        let held = FragmentSource::Fragments {
            instruction: kept.instruction,
            cell: kept.cell(),
        };
        ProductStep { lhs: held, ..*next }
            .call(device, target, OpenedBy::Accumulator)
            .map(|call| StepCost::of(call.instruction, call.cell()))
    }

    /// How a stage its operands are staged through lays its cells out, asked for as `stated` and
    /// filled by `delivery`, where this Step's sum is drained into an output: the arrangement asked
    /// for, made one the instruction computing the Step reads and the fill writes
    /// ([`RowChunks::read_by`](crate::stage::RowChunks)). What the kernel stages its operands as,
    /// and what the host sizes the stage by ([`StageStorage::bytes`]). `None` where the Lowering
    /// cannot compute the Step on `device` compiling for `target`.
    pub fn stage_storage(
        &self,
        stated: StageStorage,
        delivery: Delivery,
        device: &DeviceProperties,
        target: &TargetProperties,
    ) -> Option<StageStorage> {
        self.call(device, target, OpenedBy::Accumulator)
            .map(|call| stated.read_by(call.instruction, delivery))
    }

    /// The Lowering's call for this Step, opened by `opened_by`.
    fn call(
        &self,
        device: &DeviceProperties,
        target: &TargetProperties,
        opened_by: OpenedBy,
    ) -> Option<ShapedInstruction> {
        Instruction::for_plane(
            device,
            target,
            self.holder,
            self.lhs,
            self.rhs,
            self.sum,
            self.shape,
            opened_by,
        )
    }
}

impl StepCost {
    /// The cost of a Step computed by `instruction` in calls filling `cell`: a cmma fragment hides
    /// which unit holds a cell, so its sum bounces through shared memory a cell at a time.
    fn of(instruction: Instruction, cell: [usize; 2]) -> Self {
        let bounced_cell = match instruction {
            Instruction::Cmma => Some(cell),
            Instruction::Registers { .. } | Instruction::Mma { .. } | Instruction::Wgmma => None,
        };
        Self { bounced_cell }
    }

    /// Cells of shared memory a drain through `scratch` takes, of a sum over `out`, the `m × n` a
    /// Scope holds: none, one call's cell, or the whole output; none where nothing bounces.
    pub fn scratch_cells(self, out: [usize; 2], scratch: Scratch) -> usize {
        let Some([m, n]) = self.bounced_cell else {
            return 0;
        };
        match scratch {
            Scratch::None => 0,
            Scratch::OneTile => m * n,
            Scratch::WholeGrid => out[0] * out[1],
        }
    }
}

/// Whether `config` contracts `lhs` by `rhs` summed in `sum`, on values that mean something
/// alone: an unscaled `e2m1` value means nothing without its block's scale.
fn products(config: &MmaConfig, lhs: ElemType, rhs: ElemType, sum: ElemType) -> bool {
    let e2m1 = ElemType::Float(FloatKind::E2M1);
    config.a_type == lhs
        && config.b_type == rhs
        && config.cd_type == sum
        && lhs != e2m1
        && rhs != e2m1
}

/// The Step's shape, `16x16x16`, as autotune names a plane by it.
impl core::fmt::Display for ProductStep {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [m, n, k] = self.shape;
        write!(f, "{m}x{n}x{k}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dimension;
    use crate::ops::matmul::instruction::tests::{
        F32, HALF, cuda, metal, plain, shape, sum_is_next_a, sum_is_not_next_a,
    };
    use crate::stage::RowChunks;

    fn sizes(device: &DeviceProperties) -> Vec<[usize; 3]> {
        ProductStep::sizes(device, ComputeScope::Plane, HALF, HALF, F32)
    }

    fn step(shape: [usize; 3]) -> ProductStep {
        ProductStep {
            shape,
            holder: ComputeScope::Plane,
            lhs: HALF,
            rhs: HALF,
            sum: F32,
        }
    }

    /// An M2 allows one Step, `8 × 8 × 8`; an `sm_80` every shape it offers, the most contracted
    /// first, then the squarest; an RDNA part its one `16 × 16 × 16`; a device with no matrix
    /// instruction none, and a plane group none here.
    #[test]
    fn a_device_allows_the_steps_of_its_instructions() {
        assert_eq!(sizes(&metal()), vec![[8, 8, 8]]);
        assert_eq!(
            sizes(&cuda()),
            vec![[16, 16, 16], [8, 32, 16], [32, 8, 16], [16, 8, 16]]
        );
        let mut rdna = plain();
        rdna.features.matmul.cmma.insert(shape(16, 16, 16));
        assert_eq!(sizes(&rdna), vec![[16, 16, 16]]);
        assert_eq!(sizes(&plain()), Vec::<[usize; 3]>::new());
        assert_eq!(
            ProductStep::sizes(&cuda(), ComputeScope::MMA_PLANE_GROUP, HALF, HALF, F32),
            Vec::<[usize; 3]>::new()
        );
    }

    /// A drained sum through the fragment API bounces through shared memory a call's cell at a
    /// time; through the crate's own transport each unit writes its cells. A Step two calls fill
    /// costs the same as one. No Step is computed where the device offers nothing.
    #[test]
    fn a_step_says_what_its_drain_spends() {
        let target = sum_is_next_a();
        let metal = step([8, 8, 8]).cost(&metal(), &target);
        assert_eq!(
            metal,
            Some(StepCost {
                bounced_cell: Some([8, 8])
            })
        );
        let wide = step([16, 16, 16]).cost(&cuda(), &target);
        assert_eq!(
            wide,
            Some(StepCost {
                bounced_cell: Some([16, 16])
            })
        );
        let mut mma_only = plain();
        mma_only.features.matmul.mma.insert(shape(16, 8, 16));
        let two_calls = step([16, 16, 16]).cost(&mma_only, &target);
        assert_eq!(two_calls, Some(StepCost { bounced_cell: None }));
        assert_eq!(step([16, 16, 16]).cost(&plain(), &target), None);
    }

    /// A sum stays in its registers for the next product where the crate's transport computes its
    /// Step and its `A` lies like its accumulator: on an `sm_80` a `16 × 8 × 16` Step, which only
    /// the crate's transport offers, not a `16 × 16 × 16` one a single cmma call fills; not on an
    /// M2, whose one `8 × 8 × 8` both offer, nor on a target laying its `A` otherwise.
    #[test]
    fn a_kept_sum_stays_in_registers_only_where_its_a_lies_alike() {
        let narrow = step([16, 8, 16]);
        assert_eq!(
            narrow.kept_for(&narrow, &cuda(), &sum_is_next_a()),
            Some(StepCost { bounced_cell: None })
        );
        let wide = step([16, 16, 16]);
        assert_eq!(wide.kept_for(&wide, &cuda(), &sum_is_next_a()), None);
        let metal_step = step([8, 8, 8]);
        assert_eq!(
            metal_step.kept_for(&metal_step, &metal(), &sum_is_next_a()),
            None
        );
        assert_eq!(
            narrow.kept_for(&narrow, &cuda(), &sum_is_not_next_a()),
            None
        );
    }

    /// A stage asked to swizzle its rows is padded where the fragment API computes the Step (an
    /// M2's `8 × 8 × 8`, an `sm_80`'s `16 × 16 × 16`), and stays swizzled where the crate's
    /// transport does (an `sm_80`'s `16 × 8 × 16`); a stage a TMA box fills lands its rows in order
    /// rather than padded, swizzled ones as they are. A stage of lines is no tiled one and stays.
    #[test]
    fn a_stage_is_arranged_as_its_step_reads_it() {
        let target = sum_is_next_a();
        let tiled = |chunks| StageStorage::Tiled {
            block: vec![(Dimension(0), 16), (Dimension(1), 16), (Dimension(2), 32)],
            chunks,
        };
        let staged = |step: ProductStep, device: &DeviceProperties, chunks, delivery| {
            step.stage_storage(tiled(chunks), delivery, device, &target)
        };
        let sync = Delivery::SyncPerUnit;
        assert_eq!(
            staged(step([8, 8, 8]), &metal(), RowChunks::Swizzled, sync),
            Some(tiled(RowChunks::Padded))
        );
        assert_eq!(
            staged(step([16, 16, 16]), &cuda(), RowChunks::Swizzled, sync),
            Some(tiled(RowChunks::Padded))
        );
        assert_eq!(
            staged(step([16, 8, 16]), &cuda(), RowChunks::Swizzled, sync),
            Some(tiled(RowChunks::Swizzled))
        );
        assert_eq!(
            staged(
                step([16, 16, 16]),
                &cuda(),
                RowChunks::Swizzled,
                Delivery::Tma
            ),
            Some(tiled(RowChunks::InOrder))
        );
        assert_eq!(
            staged(
                step([16, 8, 16]),
                &cuda(),
                RowChunks::Swizzled,
                Delivery::Tma
            ),
            Some(tiled(RowChunks::Swizzled))
        );
        assert_eq!(
            staged(step([16, 8, 16]), &cuda(), RowChunks::Padded, Delivery::Tma),
            Some(tiled(RowChunks::InOrder))
        );
        assert_eq!(
            staged(step([8, 8, 8]), &metal(), RowChunks::InOrder, sync),
            Some(tiled(RowChunks::InOrder))
        );
        assert_eq!(
            step([8, 8, 8]).stage_storage(StageStorage::Strided, sync, &metal(), &target),
            Some(StageStorage::Strided)
        );
        assert_eq!(
            staged(step([16, 16, 16]), &plain(), RowChunks::Swizzled, sync),
            None
        );
    }

    /// A bounced drain takes no cells without scratch, one call's cell at a time, or the whole
    /// output at once; a sum written cell by cell takes none.
    #[test]
    fn scratch_takes_a_cell_or_the_whole_output() {
        let bounced = StepCost {
            bounced_cell: Some([16, 8]),
        };
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::None), 0);
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::OneTile), 128);
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::WholeGrid), 1024);
        let written = StepCost { bounced_cell: None };
        assert_eq!(written.scratch_cells([32, 32], Scratch::WholeGrid), 0);
    }

    #[test]
    fn a_step_is_named_by_its_shape() {
        assert_eq!(step([16, 8, 16]).to_string(), "16x8x16");
    }
}
