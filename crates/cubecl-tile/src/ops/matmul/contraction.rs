//! What a product multiplies and what its sum is for ([`Contraction`]), the matrix instruction a
//! device contracts it through ([`Contraction::instruction`]), and the smallest box everything
//! touching it handles whole ([`Contraction::step`]).
//!
//! The choice has one right answer on a device, so it is made here, once, from facts the host
//! and the kernel both hold, and never raced.

use core::cmp::Reverse;

use cubecl::ir::features::{MmaConfig, Plane};
use cubecl::ir::{DeviceProperties, ElemType, TargetProperties};

use crate::{Axis, Instruction, MmaIo, MmaOperands, OperandReads, Space};

/// What becomes of a sum once its product is contracted: structure the code states where it
/// opens the sum, never a setting.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SumRole {
    /// Written out: drained into an output, or copied to shared memory.
    Drained,
    /// Kept in its registers as the `A` of a later product, as an attention keeps its score.
    InRegisters,
}

/// One matrix product as the tile layer sees it: its two inputs, the element its sum is held in,
/// and what becomes of that sum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Contraction {
    pub lhs: Input,
    pub rhs: Input,
    pub sum: ElemType,
    pub role: SumRole,
}

/// One input of a product: the element the instruction contracts it in, and how it is read as
/// stored ([`OperandReads`]): one value a read for a float input, whole words and words of scales
/// for a packed one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Input {
    pub element: ElemType,
    pub reads: OperandReads,
}

impl Input {
    /// An input read as plain values of `element`.
    pub fn plain(element: ElemType) -> Self {
        Self {
            element,
            reads: OperandReads::plain(),
        }
    }
}

/// A matrix instruction and the shape of the fragment one call of it contracts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fragment {
    pub instruction: Instruction,
    pub shape: MmaConfig,
}

impl Contraction {
    /// The matrix instruction `device` contracts this product through, compiling for `target`,
    /// and the fragment one call contracts. `None` where the device offers none for these
    /// elements, or its planes have no single width a fragment can be laid over.
    ///
    /// The rule:
    /// - A sum kept in registers, on a target that lays `A` out like its accumulator: `mma` first,
    ///   so the sum is the next product's `A` where it lies.
    /// - Otherwise `cmma` first.
    /// - Within a transport, the shape contracting the most per call, then the squarest: a square
    ///   shape reads an input laid along `k` as well as one laid along `n`.
    pub fn instruction(
        &self,
        device: &DeviceProperties,
        target: &TargetProperties,
    ) -> Option<Fragment> {
        let hardware = &device.hardware;
        let planes_of_one_width = device.features.plane.contains(Plane::Ops)
            && hardware.plane_size_min == hardware.plane_size_max;
        if !planes_of_one_width {
            return None;
        }

        let layout = &target.mma;
        let accumulator_is_next_a = layout.register_layout_a == layout.register_layout_acc
            && layout.register_duplication_a == layout.register_duplication_acc;
        let offered = &device.features.matmul;
        let cmma = (&offered.cmma, Instruction::Cmma);
        let mma = (
            &offered.mma,
            Instruction::Mma {
                io: MmaIo::default(),
                operands: MmaOperands::Plain,
            },
        );
        let transports = match self.role == SumRole::InRegisters && accumulator_is_next_a {
            true => [mma, cmma],
            false => [cmma, mma],
        };

        transports.into_iter().find_map(|(shapes, instruction)| {
            shapes
                .iter()
                .filter(|s| {
                    s.a_type == self.lhs.element
                        && s.b_type == self.rhs.element
                        && s.cd_type == self.sum
                })
                .min_by_key(|s| (Reverse(s.m * s.n * s.k), s.m.abs_diff(s.n), **s))
                .map(|&shape| Fragment { instruction, shape })
        })
    }

    /// The smallest box over `axes` (`m`, `n`, `k`) that everything touching this product handles
    /// whole: one call of its instruction ([`instruction`](Self::instruction)), every read of a
    /// packed input, of its values and of its scales, and every load of a plain input, a line of
    /// the device's load width along the axis the other input's reads run along (`k` where the
    /// other is plain). The leaf a plane's share of the product is cut to; for plain inputs, the
    /// instruction's fragment. `None` where the device has no instruction for it.
    ///
    /// The lhs spans `m` and `k`, the rhs `k` and `n`.
    pub fn step(
        &self,
        device: &DeviceProperties,
        target: &TargetProperties,
        [m, n, k]: [Axis; 3],
    ) -> Option<Space> {
        let shape = self.instruction(device, target)?.shape;
        let mut parts = vec![Space::new(&[
            (m, shape.m as usize),
            (n, shape.n as usize),
            (k, shape.k as usize),
        ])];
        let inputs = [
            (&self.lhs, [m, k], &self.rhs),
            (&self.rhs, [k, n], &self.lhs),
        ];
        for (input, spans, other) in inputs {
            if !input.reads.is_plain() {
                parts.push(input.reads.read_space());
                parts.extend(input.reads.scale_read_space());
                continue;
            }
            // The other input's reads run along the axis its words extend over, `k` for a plain
            // one, whose loads are lines down `k`.
            let along = other.reads.read.axis().unwrap_or(k);
            if spans.contains(&along) {
                let bits = input.element.size_bits().max(1);
                let line = (device.hardware.load_width as usize / bits).max(1);
                parts.push(Space::new(&[(along, line)]));
            }
        }
        Some(Space::covering(&parts.iter().collect::<Vec<_>>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MicroTile;
    use cubecl::cmma::MatrixLayout;
    use cubecl::ir::{
        DeviceIdentity, FloatKind, HardwareProperties, MemoryDeviceProperties, VectorSize,
        features::Features,
    };
    use cubecl::profile::TimingMethod;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    fn half(role: SumRole) -> Contraction {
        Contraction {
            lhs: Input::plain(F16),
            rhs: Input::plain(F16),
            sum: F32,
            role,
        }
    }

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
        device.features.matmul.mma.insert(shape(16, 8, 8));
        device
    }

    /// A target whose `A` and accumulator lie alike, as on CUDA and Metal.
    fn accumulator_like_a() -> TargetProperties {
        let mut target = TargetProperties::default();
        target.mma.register_layout_a = target.mma.register_layout_acc;
        target.mma.register_duplication_a = target.mma.register_duplication_acc;
        target
    }

    /// A target whose `A` lies otherwise than its accumulator.
    fn accumulator_unlike_a() -> TargetProperties {
        let mut target = accumulator_like_a();
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

    /// A sum written out goes through the vendor's fragment API, at its square shape.
    #[test]
    fn a_drained_sum_goes_through_cmma() {
        for (device, want) in [(metal(), shape(8, 8, 8)), (cuda(), shape(16, 16, 16))] {
            let fragment = half(SumRole::Drained).instruction(&device, &accumulator_like_a());
            assert_eq!(
                fragment,
                Some(Fragment {
                    instruction: Instruction::Cmma,
                    shape: want
                })
            );
        }
    }

    /// A sum kept in registers goes through `mma` where its accumulator is the next `A`, at the
    /// deepest shape.
    #[test]
    fn a_sum_kept_in_registers_goes_through_mma() {
        for (device, want) in [(metal(), shape(8, 8, 8)), (cuda(), shape(16, 8, 16))] {
            let fragment = half(SumRole::InRegisters).instruction(&device, &accumulator_like_a());
            assert_eq!(
                fragment,
                Some(Fragment {
                    instruction: mma(),
                    shape: want
                })
            );
        }
    }

    /// Where the accumulator is not the next `A`, keeping the sum in registers buys nothing.
    #[test]
    fn a_sum_kept_in_registers_needs_its_accumulator_as_a() {
        let fragment = half(SumRole::InRegisters).instruction(&cuda(), &accumulator_unlike_a());
        assert_eq!(fragment.map(|f| f.instruction), Some(Instruction::Cmma));
    }

    /// A device offering only one transport is reached through it, whatever the role.
    #[test]
    fn one_transport_serves_every_role() {
        let mut device = plain();
        device.features.matmul.mma.insert(shape(16, 8, 16));
        for role in [SumRole::Drained, SumRole::InRegisters] {
            let fragment = half(role).instruction(&device, &accumulator_like_a());
            assert_eq!(fragment.map(|f| f.instruction), Some(mma()));
        }
    }

    /// No instruction for the elements, or planes of no single width: none.
    #[test]
    fn no_instruction_where_none_fits() {
        let other = Contraction {
            lhs: Input::plain(F32),
            rhs: Input::plain(F32),
            sum: F32,
            role: SumRole::Drained,
        };
        assert_eq!(other.instruction(&cuda(), &accumulator_like_a()), None);

        let mut uneven = cuda();
        uneven.hardware.plane_size_min = 16;
        assert_eq!(
            half(SumRole::Drained).instruction(&uneven, &accumulator_like_a()),
            None
        );
    }

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// Plain inputs: the step is the instruction's fragment, every line of an input fitting in
    /// it.
    #[test]
    fn a_plain_step_is_the_fragment() {
        let target = accumulator_like_a();
        let step = |device| half(SumRole::Drained).step(&device, &target, [M, N, K]);
        assert_eq!(step(metal()), Some(Space::new(&[(M, 8), (N, 8), (K, 8)])));
        assert_eq!(step(cuda()), Some(Space::new(&[(M, 16), (N, 16), (K, 16)])));
    }

    /// NVFP4 rows against an `f16` activation on an `8 × 8 × 8` device: a read of values is
    /// sixteen of `k`, a read of scales a word of four, sixty-four of `k`, so the step is that
    /// deep, the `f16` activation loaded eight of `k` at a time eight times.
    #[test]
    fn a_packed_step_is_a_word_of_scales_deep() {
        let nvfp4 = Contraction {
            rhs: Input {
                element: F16,
                reads: OperandReads::packed(K, 16, K, 16, 4),
            },
            ..half(SumRole::Drained)
        };
        let step = nvfp4.step(&metal(), &accumulator_like_a(), [M, N, K]);
        assert_eq!(step, Some(Space::new(&[(M, 8), (N, 8), (K, 64)])));
    }

    /// A plain input whose loads run along an axis it lacks adds nothing: the lhs meets an rhs
    /// packed along `n` only through the instruction.
    #[test]
    fn a_line_along_an_axis_the_input_lacks_adds_nothing() {
        let along_n = Contraction {
            rhs: Input {
                element: F16,
                reads: OperandReads {
                    read: MicroTile::along(N, 16),
                    ..OperandReads::plain()
                },
            },
            ..half(SumRole::Drained)
        };
        let step = along_n.step(&metal(), &accumulator_like_a(), [M, N, K]);
        assert_eq!(step, Some(Space::new(&[(M, 8), (N, 16), (K, 8)])));
    }
}
