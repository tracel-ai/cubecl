//! What an NVIDIA compute capability is, for every backend that generates code for one.
//!
//! The CUDA runtime drives two of them, and the `-arch` a kernel is built for has to be the
//! same answer whichever one compiles it. The counterpart of [`amd`](crate::amd), kept
//! separate because the two name their devices nothing alike: AMD has a gfx string the driver
//! reports verbatim, NVIDIA a pair of integers.

use crate::{
    CoordinatePart, ElementIndex, ElementPositions,
    types::matrix::{MatrixIdent, MatrixType},
};
use alloc::{format, string::String, vec};
use pliron::context::Context;

/// The bits each register of an `mma.sync` fragment holds.
const REGISTER_BITS: usize = 32;

/// Where the elements a lane holds of `matrix` lie in it, in the manual `mma.sync` matrix API.
///
/// NVIDIA documents that mapping, and every backend generating `mma.sync` has to agree with it
/// exactly -- a kernel that indexes its own fragment differently from the way the instruction
/// reads it computes a wrong answer rather than failing -- so it is written once here.
///
/// Derived from the PTX shape documentation:
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
pub fn element_positions(ctx: &Context, matrix: &MatrixType) -> ElementPositions {
    let register = REGISTER_BITS / matrix.unpacked_elem_size_bits(ctx);
    let (lane, nth) = (
        CoordinatePart::of(ElementIndex::Lane),
        CoordinatePart::of(ElementIndex::Nth),
    );
    // The lanes come in groups of four, and their registers in pairs.
    let group = lane.in_runs_of(4);
    let lane_of_group = lane.counted_over(4);
    let of_register = nth.counted_over(register);
    let of_pair = |part: CoordinatePart| part.counted_over(2);
    match matrix.ident {
        MatrixIdent::A => ElementPositions {
            row: vec![group, of_pair(nth.in_runs_of(register)).spaced_by(8)],
            col: vec![
                lane_of_group.spaced_by(register),
                of_register,
                of_pair(nth.in_runs_of(2 * register)).spaced_by(4 * register),
            ],
        },
        MatrixIdent::B => ElementPositions {
            row: vec![
                lane_of_group.spaced_by(register),
                of_register,
                nth.in_runs_of(register).spaced_by(4 * register),
            ],
            col: vec![group],
        },
        MatrixIdent::Accumulator => ElementPositions {
            row: vec![group, of_pair(nth.in_runs_of(2)).spaced_by(8)],
            col: vec![lane_of_group.spaced_by(2), of_pair(nth)],
        },
    }
}

/// A compute capability, as `cuDeviceGetAttribute` reports it.
///
/// Held as the packed two-digit form the CUDA tooling is written in — 8.6 is `86` — because
/// that is what `sm_86` and every threshold in the feature tables are spelled with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmArch {
    version: u32,
    /// Whether the die has tensor cores. Compute capability cannot say this on its own: the
    /// GTX-branded Turings report 7.5 like every other Turing and have none. Carried here so
    /// both backends read one answer rather than each deciding.
    tensor_cores: bool,
    /// Whether the architecture-specific (`a`) variant is the one to target. Blackwell and
    /// Hopper gate their tensor-core instructions behind it, and a `sm_90` kernel cannot
    /// reach them.
    arch_specific: bool,
}

impl SmArch {
    /// The architecture the runtime probed, with `tensor_cores` as it decided.
    pub fn new(version: u32, tensor_cores: bool) -> Self {
        Self {
            version,
            tensor_cores,
            // The same threshold the C++ backend's `--gpu-architecture` uses.
            arch_specific: version >= 90,
        }
    }

    /// The packed two-digit capability: 8.6 is `86`.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Whether the die has tensor cores at all.
    pub fn tensor_cores(&self) -> bool {
        self.tensor_cores
    }

    /// The `-mcpu` the NVPTX target machine takes, and the `.target` directive the emitted
    /// PTX carries — `sm_86`, or `sm_90a` where the architecture-specific variant exists.
    pub fn target_cpu(&self) -> String {
        if self.arch_specific {
            format!("sm_{}a", self.version)
        } else {
            format!("sm_{}", self.version)
        }
    }

    /// A warp is 32 lanes on every CUDA device there has ever been, which is why this is a
    /// constant here where the AMD side has to consult a table.
    pub fn plane_dim(&self) -> u32 {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ElemType, FloatKind,
        types::matrix::{MatrixLayout, MatrixScope, MatrixShape},
    };
    use alloc::string::ToString;

    /// Every cell of an operand is held once, whatever its elements' width.
    #[test]
    fn the_lanes_of_a_plane_hold_an_operand_together() {
        const LANES: usize = 32;
        let ctx = Context::default();
        for (elem, shape) in [
            (FloatKind::F32, (16, 8, 8)),
            (FloatKind::F16, (16, 8, 16)),
            (FloatKind::E4M3, (16, 8, 32)),
        ] {
            let shape = MatrixShape::from(shape);
            for ident in [MatrixIdent::A, MatrixIdent::B, MatrixIdent::Accumulator] {
                let matrix = MatrixType {
                    ident,
                    shape,
                    elem_ty: ElemType::Float(elem).to_type(&ctx),
                    layout: MatrixLayout::RowMajor,
                    scope: MatrixScope::Plane,
                };
                let cells = shape.num_elems(ident);
                let held = element_positions(&ctx, &matrix).held_cells(
                    (shape, ident),
                    LANES,
                    cells / LANES,
                );
                assert_eq!(held, vec![1; cells], "{elem:?} {ident}");
            }
        }
    }

    /// Hopper and Blackwell need the `a` suffix to reach their tensor-core instructions;
    /// everything before it must not carry one, since `sm_86a` is not a target that exists.
    #[test]
    fn only_hopper_and_later_ask_for_the_architecture_specific_variant() {
        assert_eq!(SmArch::new(86, true).target_cpu(), "sm_86".to_string());
        assert_eq!(SmArch::new(89, true).target_cpu(), "sm_89".to_string());
        assert_eq!(SmArch::new(90, true).target_cpu(), "sm_90a".to_string());
        assert_eq!(SmArch::new(120, true).target_cpu(), "sm_120a".to_string());
    }

    /// The tensor-core answer is the runtime's, not the version's: two dies reporting 7.5
    /// differ on it, and a backend that re-derived it from the version would disagree with
    /// the features the runtime advertised.
    #[test]
    fn the_tensor_core_answer_is_carried_not_rederived() {
        assert!(!SmArch::new(75, false).tensor_cores());
        assert!(SmArch::new(75, true).tensor_cores());
    }
}
