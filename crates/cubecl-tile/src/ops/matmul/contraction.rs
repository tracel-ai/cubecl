//! What a product multiplies and what its sum is for ([`Contraction`]), and the matrix
//! instruction a device contracts it through ([`Contraction::instruction`]).
//!
//! The choice has one right answer on a device, so it is made here, once, from facts the host
//! and the kernel both hold, and never raced.

use core::cmp::Reverse;

use cubecl::ir::features::{MmaConfig, Plane};
use cubecl::ir::{DeviceProperties, ElemType, MmaProperties, TargetProperties};

use crate::{Instruction, MmaIo, MmaOperands};

/// What becomes of a sum once its product is contracted: structure the code states where it
/// opens the sum, never a setting.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SumRole {
    /// Written out: drained into an output, or copied to shared memory.
    Drained,
    /// Kept in its registers as the `A` of a later product, as an attention keeps its score.
    InRegisters,
}

/// One matrix product as the tile layer sees it: the element of each input, the element its sum
/// is held in, and what becomes of that sum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Contraction {
    pub lhs: ElemType,
    pub rhs: ElemType,
    pub sum: ElemType,
    pub role: SumRole,
}

/// A matrix instruction and the shape of the fragment one call of it contracts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fragment {
    pub instruction: Instruction,
    pub shape: MmaConfig,
}

/// The two ways a plane reaches a device's matrix instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Transport {
    /// The vendor's fragment API (`wmma`, `simdgroup_matrix`): the fragment's registers are
    /// hidden.
    Cmma,
    /// This crate's own transport over the matrix intrinsics: each unit holds known cells.
    Mma,
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
    /// - Within a transport, the shape contracting the most per call, then the squarest.
    pub fn instruction(
        &self,
        device: &DeviceProperties,
        target: &TargetProperties,
    ) -> Option<Fragment> {
        if !planes_of_one_width(device) {
            return None;
        }
        self.transports(target).into_iter().find_map(|transport| {
            transport.largest(device, self).map(|shape| Fragment {
                instruction: transport.instruction(),
                shape,
            })
        })
    }

    /// The transports in the order this product tries them.
    fn transports(&self, target: &TargetProperties) -> [Transport; 2] {
        let in_registers = self.role == SumRole::InRegisters && accumulator_as_a(&target.mma);
        match in_registers {
            true => [Transport::Mma, Transport::Cmma],
            false => [Transport::Cmma, Transport::Mma],
        }
    }

    /// Whether `shape` multiplies these inputs into this sum.
    fn takes(&self, shape: &MmaConfig) -> bool {
        shape.a_type == self.lhs && shape.b_type == self.rhs && shape.cd_type == self.sum
    }
}

impl Transport {
    /// The shape of this transport `device` offers for `contraction` that contracts the most per
    /// call.
    fn largest(self, device: &DeviceProperties, contraction: &Contraction) -> Option<MmaConfig> {
        let offered = match self {
            Transport::Cmma => &device.features.matmul.cmma,
            Transport::Mma => &device.features.matmul.mma,
        };
        offered
            .iter()
            .filter(|shape| contraction.takes(shape))
            .copied()
            .min_by_key(size_rank)
    }

    fn instruction(self) -> Instruction {
        match self {
            Transport::Cmma => Instruction::Cmma,
            Transport::Mma => Instruction::Mma {
                io: MmaIo::default(),
                operands: MmaOperands::Plain,
            },
        }
    }
}

/// Whether `mma` lays an `A`'s cells where it lays an accumulator's, each held as often: then a
/// sum contracted again is its next product's `A` in the same registers.
fn accumulator_as_a(mma: &MmaProperties) -> bool {
    mma.register_layout_a == mma.register_layout_acc
        && mma.register_duplication_a == mma.register_duplication_acc
}

/// Whether `device`'s planes run plane operations at one width, which a fragment is laid over.
fn planes_of_one_width(device: &DeviceProperties) -> bool {
    let hardware = &device.hardware;
    device.features.plane.contains(Plane::Ops)
        && hardware.plane_size_min == hardware.plane_size_max
}

/// The order shapes are preferred in, first the least: the most contracted per call, then the
/// squarest (a square shape reads an input laid along `k` as well as one laid along `n`), then
/// the shape itself, so every shape has its own place.
#[allow(clippy::type_complexity)]
fn size_rank(shape: &MmaConfig) -> (Reverse<u64>, u32, u32, u32, u32, MmaConfig) {
    let (m, n, k) = (shape.m, shape.n, shape.k);
    (
        Reverse(u64::from(m) * u64::from(n) * u64::from(k)),
        m.abs_diff(n),
        m,
        n,
        k,
        *shape,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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
            lhs: F16,
            rhs: F16,
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
            lhs: F32,
            rhs: F32,
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
}
