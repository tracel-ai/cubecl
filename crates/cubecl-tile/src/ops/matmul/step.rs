//! What a host reads of a product's Step ([`ProductStep`]): the Steps a device allows and what
//! computing one spends ([`StepCost`]). The Lowering picks what computes a Step; a host never
//! sees it.

use core::cmp::Reverse;

use cubecl::cmma::MatrixLayout;
use cubecl::ir::features::MmaConfig;
use cubecl::ir::{DeviceProperties, ElemType, FloatKind, TargetProperties};

use super::instruction::{FragmentSource, Instruction, OpenedBy, ShapedInstruction};
use crate::launch::{Delivery, VectorRequirement};
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
///
/// # Assumptions
///
/// - A drained sum's destination is bound at a width its
///   [`line_requirement`](StepCost::line_requirement) admits.
///
/// # Guarantees
///
/// - A drained sum's cell is bounced only where the instruction computing the Step hides which
///   unit holds a cell of its sum on the target ([`Instruction::hides_cell_positions`]), and a
///   rescaled sum's only where the instruction is [`Instruction::Cmma`], whose sum a scratch
///   backs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StepCost {
    /// The cell a drained sum is spilled through shared memory in, one call's output; `None`
    /// where each unit holds its own cells and writes them where they go.
    pub bounced_cell: Option<[usize; 2]>,
    /// The widest line along the sum's rows (`m`) and along its columns (`n`) that its units
    /// write out of their registers where nothing bounces ([`Instruction::widest_line`]).
    pub widest_line: [usize; 2],
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
            .map(|call| StepCost::drained(call, target, self.sum))
    }

    /// What this Step spends where its sum's rows are scaled between its products
    /// ([`Instruction::for_rescaled_accumulator`]): nothing bounced where the sum is held in the
    /// manual mma. `None` where the device offers no matrix instruction for it.
    pub fn rescaled_cost(
        &self,
        device: &DeviceProperties,
        target: &TargetProperties,
    ) -> Option<StepCost> {
        self.call(device, target, OpenedBy::RescaledAccumulator)
            .map(StepCost::rescaled)
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
            .map(|call| StepCost::drained(call, target, next.sum))
    }

    /// How a stage its operands are staged through lays its cells out, asked for as `stated` and
    /// filled by `delivery`, where this Step's sum is drained into an output: the arrangement asked
    /// for, made one the instruction computing the Step reads and the fill writes
    /// ([`RowChunks::read_by`](crate::stage::RowChunks)). What the kernel stages its operands as,
    /// and what the host sizes the stage by ([`StageStorage::bytes`]). `None` where the Lowering
    /// cannot compute the Step on `device` compiling for `target`.
    ///
    /// # Guarantees
    ///
    /// - The storage returned is one the instruction computing the Step reads and `delivery`
    ///   writes, of the kind, block and order `stated` names.
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
}

impl ProductStep {
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

    /// What the drained sum asks of `operand`, its destination, whose lines run along the sum's
    /// rows where `layout` is row-major and along its columns where it is col-major: a width
    /// dividing the bounced cell's side where a cell bounces, which its scratch is read back in,
    /// and the [`widest_line`](StepCost::widest_line) along that side where nothing bounces. One
    /// value a line for an [`Undefined`](MatrixLayout::Undefined) layout, which names no side.
    ///
    /// # Guarantees
    ///
    /// - Where nothing bounces, every width the requirement admits is one the units write out of
    ///   their registers ([`CellLines::widest`](crate::CellLines::widest)).
    pub fn line_requirement<O>(&self, operand: O, layout: MatrixLayout) -> VectorRequirement<O> {
        let lines = self.bounced_cell.unwrap_or(self.widest_line);
        let elements = match layout {
            MatrixLayout::RowMajor => lines[1],
            MatrixLayout::ColMajor => lines[0],
            MatrixLayout::Undefined => 1,
        };
        VectorRequirement::Access { operand, elements }
    }
}

impl StepCost {
    /// The cost of a Step computed by `call` on `target` whose sum, held in `sum`, is drained: an
    /// instruction that hides which unit holds a cell there
    /// ([`Instruction::hides_cell_positions`]) bounces its sum through shared memory a call's
    /// cell at a time, and any other writes lines out of its registers
    /// ([`Instruction::widest_line`]).
    fn drained(call: ShapedInstruction, target: &TargetProperties, sum: ElemType) -> Self {
        Self {
            bounced_cell: call
                .instruction
                .hides_cell_positions(target)
                .then_some(call.cell()),
            widest_line: call.instruction.widest_line(target, call.cell(), sum),
        }
    }

    /// The cost of a Step computed by `call` whose sum's rows are scaled between its products: an
    /// instruction that backs its sum with a scratch ([`Instruction::takes_scratch`]) scales it
    /// through it, a call's cell at a time, wherever its cells' positions are stated.
    fn rescaled(call: ShapedInstruction) -> Self {
        Self {
            bounced_cell: call.instruction.takes_scratch().then_some(call.cell()),
            widest_line: [1, 1],
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
    use crate::ops::matmul::instruction::tests::{
        F32, HALF, cuda, metal, plain, shape, sum_is_next_a, sum_is_not_next_a,
    };
    use crate::stage::RowChunks;
    use crate::{Dimension, StageOrder};

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

    /// An M2 allows one Step, `8 × 8 × 8`.
    #[test]
    fn an_m2_allows_its_one_step() {
        assert_eq!(sizes(&metal()), vec![[8, 8, 8]]);
    }

    /// An `sm_80` allows every shape it offers, the most contracted first, then the squarest.
    #[test]
    fn an_sm_80_allows_its_steps_most_contracted_first() {
        assert_eq!(
            sizes(&cuda()),
            vec![[16, 16, 16], [8, 32, 16], [32, 8, 16], [16, 8, 16]]
        );
    }

    /// An RDNA part allows its one `16 × 16 × 16`.
    #[test]
    fn an_rdna_part_allows_its_one_step() {
        let mut rdna = plain();
        rdna.features.matmul.cmma.insert(shape(16, 16, 16));
        assert_eq!(sizes(&rdna), vec![[16, 16, 16]]);
    }

    /// A device with no matrix instruction allows no Step.
    #[test]
    fn a_device_without_matrix_instructions_allows_no_step() {
        assert_eq!(sizes(&plain()), Vec::<[usize; 3]>::new());
    }

    /// A plane group's Step is its own instruction's, so none is listed here.
    #[test]
    fn a_plane_group_is_allowed_no_step_here() {
        assert_eq!(
            ProductStep::sizes(&cuda(), ComputeScope::MMA_PLANE_GROUP, HALF, HALF, F32),
            Vec::<[usize; 3]>::new()
        );
    }

    /// A drained sum through the fragment API bounces through shared memory a call's cell at a
    /// time.
    #[test]
    fn a_fragment_api_step_bounces_a_cell_at_a_time() {
        let target = sum_is_next_a();
        assert_eq!(
            step([8, 8, 8]).cost(&metal(), &target),
            Some(StepCost {
                bounced_cell: Some([8, 8]),
                widest_line: [1, 1],
            })
        );
        assert_eq!(
            step([16, 16, 16]).cost(&cuda(), &target),
            Some(StepCost {
                bounced_cell: Some([16, 16]),
                widest_line: [1, 1],
            })
        );
    }

    /// Through the crate's own transport each unit writes its cells, and a Step two calls fill
    /// costs the same as one.
    #[test]
    fn a_transported_step_bounces_nothing() {
        let mut mma_only = plain();
        mma_only.features.matmul.mma.insert(shape(16, 8, 16));
        assert_eq!(
            step([16, 16, 16]).cost(&mma_only, &sum_is_next_a()),
            Some(StepCost {
                bounced_cell: None,
                widest_line: [1, 1],
            })
        );
    }

    /// No Step is computed where the device offers nothing.
    #[test]
    fn a_step_the_device_cannot_compute_has_no_cost() {
        assert_eq!(step([16, 16, 16]).cost(&plain(), &sum_is_next_a()), None);
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
            Some(StepCost {
                bounced_cell: None,
                widest_line: [1, 1],
            })
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

    fn tiled(chunks: RowChunks) -> StageStorage {
        StageStorage::Tiled {
            block: vec![(Dimension(0), 16), (Dimension(1), 16), (Dimension(2), 32)],
            chunks,
            order: StageOrder::AsSource,
        }
    }

    fn staged(
        step: ProductStep,
        device: &DeviceProperties,
        chunks: RowChunks,
        delivery: Delivery,
    ) -> Option<StageStorage> {
        step.stage_storage(tiled(chunks), delivery, device, &sum_is_next_a())
    }

    /// A stage asked to swizzle its rows is padded where the fragment API computes the Step: an
    /// M2's `8 × 8 × 8` and an `sm_80`'s `16 × 16 × 16`.
    #[test]
    fn a_fragment_api_step_pads_a_swizzled_stage() {
        let sync = Delivery::SyncPerUnit;
        assert_eq!(
            staged(step([8, 8, 8]), &metal(), RowChunks::Swizzled, sync),
            Some(tiled(RowChunks::Padded))
        );
        assert_eq!(
            staged(step([16, 16, 16]), &cuda(), RowChunks::Swizzled, sync),
            Some(tiled(RowChunks::Padded))
        );
    }

    /// A stage stays swizzled where the crate's transport computes the Step, an `sm_80`'s
    /// `16 × 8 × 16`, whoever fills it.
    #[test]
    fn a_transported_step_keeps_a_swizzled_stage() {
        for delivery in [Delivery::SyncPerUnit, Delivery::Tma] {
            assert_eq!(
                staged(step([16, 8, 16]), &cuda(), RowChunks::Swizzled, delivery),
                Some(tiled(RowChunks::Swizzled)),
                "{delivery:?}"
            );
        }
    }

    /// A stage a TMA box fills lands its rows in order rather than padded, a swizzle the fragment
    /// API cannot read included.
    #[test]
    fn a_box_lands_a_padded_stage_in_order() {
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
            staged(step([16, 8, 16]), &cuda(), RowChunks::Padded, Delivery::Tma),
            Some(tiled(RowChunks::InOrder))
        );
    }

    /// A stage in order, or no tiled one, stays as it is.
    #[test]
    fn a_stage_in_order_stays_as_it_is() {
        let sync = Delivery::SyncPerUnit;
        assert_eq!(
            staged(step([8, 8, 8]), &metal(), RowChunks::InOrder, sync),
            Some(tiled(RowChunks::InOrder))
        );
        assert_eq!(
            step([8, 8, 8]).stage_storage(StageStorage::STRIDED, sync, &metal(), &sum_is_next_a()),
            Some(StageStorage::STRIDED)
        );
    }

    /// No storage is chosen where the device cannot compute the Step.
    #[test]
    fn a_step_the_device_cannot_compute_stages_nothing() {
        assert_eq!(
            staged(
                step([16, 16, 16]),
                &plain(),
                RowChunks::Swizzled,
                Delivery::SyncPerUnit
            ),
            None
        );
    }

    /// A bounced drain takes no cells without scratch, one call's cell at a time, or the whole
    /// output at once; a sum written cell by cell takes none.
    #[test]
    fn scratch_takes_a_cell_or_the_whole_output() {
        let bounced = StepCost {
            bounced_cell: Some([16, 8]),
            widest_line: [1, 1],
        };
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::None), 0);
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::OneTile), 128);
        assert_eq!(bounced.scratch_cells([32, 32], Scratch::WholeGrid), 1024);
        let written = StepCost {
            bounced_cell: None,
            widest_line: [1, 1],
        };
        assert_eq!(written.scratch_cells([32, 32], Scratch::WholeGrid), 0);
    }

    /// RDNA3's target, its fragments' positions stated as its LLVM compiler states them.
    fn rdna3_stated() -> TargetProperties {
        use cubecl::ir::FragmentPositions;
        use cubecl_ir::amd::{AmdMatrixCore, AmdWmma};
        let mut mma = AmdMatrixCore::Wmma(AmdWmma::Rdna3).mma_properties();
        mma.fragment_positions = FragmentPositions::AsElementPositions;
        TargetProperties { mma }
    }

    /// A device offering the `16 × 16 × 16` fragment alone.
    fn rdna() -> DeviceProperties {
        let mut rdna = plain();
        rdna.features.matmul.cmma.insert(shape(16, 16, 16));
        rdna
    }

    /// On a target that states its fragments' positions, a drained cmma sum bounces nothing and
    /// writes the lines its lanes hold, while a rescaled one still scales through its scratch.
    #[test]
    fn a_stated_fragment_is_drained_from_registers_but_rescaled_through_its_scratch() {
        let target = rdna3_stated();
        let sum = step([16, 16, 16]);
        assert_eq!(
            sum.cost(&rdna(), &target),
            Some(StepCost {
                bounced_cell: None,
                widest_line: [1, 8],
            })
        );
        assert_eq!(
            sum.rescaled_cost(&rdna(), &target),
            Some(StepCost {
                bounced_cell: Some([16, 16]),
                widest_line: [1, 1],
            })
        );
    }

    /// A sum written out of registers asks for lines dividing its widest along the side the
    /// destination's lines run along, and one value a line for a layout that names no side.
    #[test]
    fn line_requirement_follows_the_destination_layout() {
        let written = StepCost {
            bounced_cell: None,
            widest_line: [2, 8],
        };
        let requirement = |layout| written.line_requirement(0usize, layout);
        let access = |elements| VectorRequirement::Access {
            operand: 0usize,
            elements,
        };
        assert_eq!(requirement(MatrixLayout::RowMajor), access(8));
        assert_eq!(requirement(MatrixLayout::ColMajor), access(2));
        assert_eq!(requirement(MatrixLayout::Undefined), access(1));
    }

    /// A bounced sum is read back out of its scratch in lines that divide its cell's side.
    #[test]
    fn a_bounced_sum_asks_for_lines_dividing_its_cell() {
        let bounced = StepCost {
            bounced_cell: Some([16, 8]),
            widest_line: [1, 1],
        };
        assert_eq!(
            bounced.line_requirement(0usize, MatrixLayout::RowMajor),
            VectorRequirement::Access {
                operand: 0usize,
                elements: 8
            }
        );
    }

    #[test]
    fn a_step_is_named_by_its_shape() {
        assert_eq!(step([16, 8, 16]).to_string(), "16x8x16");
    }
}
