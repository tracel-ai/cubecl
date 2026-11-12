//! How an operand's bytes reach a stage ([`Delivery`]): who issues each move ([`scope`]), the order
//! its units walk the stage's lines in ([`order`]), and when what it moved has landed
//! ([`landing`]).
//!
//! [`scope`]: Delivery::scope
//! [`order`]: Delivery::order
//! [`landing`]: Delivery::landing

use cubecl::ir::{DeviceProperties, OpaqueType, features::Tma};

use crate::stage::Prefetch;
use crate::{ComputeScope, Refusal, Rendezvous, Space};

/// How an operand's bytes reach a stage filled from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    /// Every unit loads its own lines and stores them, through its registers: landed when the fill
    /// returns, and a line can be decoded on its way.
    #[default]
    SyncPerUnit,
    /// Every unit hands its own lines to the copy engine (`cp.async`), memory to memory: they land
    /// after the fill returns, as they lie, and something the slot waits on tracks them.
    AsyncPerUnit,
    /// One unit hands the stage's box to the TMA engine through a tensor map
    /// (`cp.async.bulk.tensor`): it lands after the fill returns, counted in bytes on the slot's
    /// barrier.
    Tma,
    /// Every unit computes its values from their coordinates: nothing is read, and the stage
    /// holds them when the fill returns.
    Procedural,
}

/// When what a delivery moved has landed in its stage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Landing {
    /// When the fill returns: the units wrote the stage themselves.
    OnReturn,
    /// After the fill returns, once each unit's committed copies arrive on the slot's barrier.
    Arrivals,
    /// After the fill returns, once the bytes the issuer announced arrive on the slot's barrier,
    /// through the async proxy.
    Bytes,
}

/// The order the units of a fill walk its lines in: line `i` of a copy is the `i`-th of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LineOrder {
    /// The destination's own order: neighbouring units write neighbouring stage lines. A
    /// delivery through the units' registers reads through the cache, which gathers the lines a
    /// warp reads from several rows back into whole transactions.
    Destination,
    /// The source's order: neighbouring units read neighbouring source lines. A delivery that
    /// bypasses the cache reads exactly the transactions its units' lines fall in, so it reads
    /// the source as it lies.
    Source,
}

/// CUDA's cap on each TMA box dimension.
const TMA_MAX_BOX_DIM: usize = 256;

/// The lengths of one copy the copy engine makes, in bytes: `cp.async` moves 4, 8 or 16.
pub(crate) const COPY_ENGINE_LINE_BYTES: [usize; 3] = [4, 8, 16];

impl Delivery {
    /// The byte alignment a stage a [`Tma`](Delivery::Tma) box lands in starts on: the span a
    /// 128-byte swizzle repeats over, which the engine keys off the address, so a stage's first
    /// row takes the first key as the fragment reading it does. A launch sizing its shared memory
    /// counts what aligning a stage to it can cost.
    pub const TMA_STAGE_ALIGNMENT: usize = 1024;

    /// Whether this delivery moves `stage` in one transfer, and why not where it cannot.
    pub fn moves(self, stage: &Space) -> Result<(), Refusal> {
        match self {
            Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::Procedural => Ok(()),
            Delivery::Tma => match stage
                .extents()
                .into_iter()
                .find(|&(_, edge)| edge > TMA_MAX_BOX_DIM)
            {
                Some((axis, edge)) => Err(Refusal::BoxPastDescriptor {
                    axis,
                    edge,
                    most: TMA_MAX_BOX_DIM,
                }),
                None => Ok(()),
            },
        }
    }

    /// Whether `device` has what this delivery runs on: the copy engine or the TMA engine, and
    /// the barrier their copies land on.
    pub fn runs_on(self, device: &DeviceProperties) -> Result<(), Refusal> {
        let barriers = device.features.types.opaque.contains(&OpaqueType::Barrier);
        let engine = match self {
            Delivery::SyncPerUnit | Delivery::Procedural => return Ok(()),
            Delivery::AsyncPerUnit => device.features.copy_async,
            Delivery::Tma => device.features.tma.contains(Tma::Base),
        };
        match engine && barriers {
            true => Ok(()),
            false => Err(Refusal::DeliveryNotOnDevice(self)),
        }
    }

    /// Whether this delivery fills a ring that holds its next stage as `prefetch` says: a stage
    /// held in the units' registers is one the units load and store themselves.
    pub fn holds(self, prefetch: Prefetch) -> Result<(), Refusal> {
        match (prefetch, self.landing()) {
            (Prefetch::InSlots, _) | (Prefetch::InRegisters, Landing::OnReturn) => Ok(()),
            (Prefetch::InRegisters, Landing::Arrivals | Landing::Bytes) => {
                Err(Refusal::PrefetchPastDelivery(self))
            }
        }
    }

    /// Whether this delivery moves lines `bits` long whole: the copy engine moves 4, 8 or 16
    /// bytes at once ([`COPY_ENGINE_LINE_BYTES`]).
    pub fn moves_lines(self, bits: usize) -> Result<(), Refusal> {
        match self {
            Delivery::AsyncPerUnit if !COPY_ENGINE_LINE_BYTES.iter().any(|&b| b * 8 == bits) => {
                Err(Refusal::LineNotCopied { bits })
            }
            Delivery::SyncPerUnit
            | Delivery::AsyncPerUnit
            | Delivery::Tma
            | Delivery::Procedural => Ok(()),
        }
    }

    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// Who issues one move: each unit its own lines ([`Unit`](ComputeScope::Unit)), or one elected
    /// unit the whole stage of the cube that owns it ([`Cube`](ComputeScope::Cube)).
    pub(crate) fn scope(self) -> ComputeScope {
        match self {
            Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::Procedural => {
                ComputeScope::Unit
            }
            Delivery::Tma => ComputeScope::Cube,
        }
    }

    /// The order the units walk the stage's lines in, where each unit issues its own; `None` for
    /// a delivery one unit issues as a box.
    pub(crate) fn order(self) -> Option<LineOrder> {
        match self {
            Delivery::SyncPerUnit | Delivery::Procedural => Some(LineOrder::Destination),
            Delivery::AsyncPerUnit => Some(LineOrder::Source),
            Delivery::Tma => None,
        }
    }

    /// When what this delivery moved has landed in the stage.
    pub(crate) fn landing(self) -> Landing {
        match self {
            Delivery::SyncPerUnit | Delivery::Procedural => Landing::OnReturn,
            Delivery::AsyncPerUnit => Landing::Arrivals,
            Delivery::Tma => Landing::Bytes,
        }
    }

    /// The synchronization required to materialize this source in a staging slot: the cube's for
    /// a stage the units wrote themselves, an mbarrier for any other.
    pub(crate) fn rendezvous(&self) -> Rendezvous {
        match self.landing() {
            Landing::OnReturn => Rendezvous::Cube,
            Landing::Arrivals | Landing::Bytes => Rendezvous::Barrier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{
        DeviceIdentity, HardwareProperties, MemoryDeviceProperties, VectorSize, features::Features,
    };
    use cubecl::profile::TimingMethod;

    /// A device with no copy engine, no TMA and no barrier type.
    fn bare() -> DeviceProperties {
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
        DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(u64::MAX, 256),
            hardware,
            TimingMethod::Device,
            DeviceIdentity {
                name: "test".to_string(),
                fingerprint: "test".to_string(),
                physical: None,
            },
        )
    }

    /// [`bare`] with both engines and the barrier their copies land on.
    fn engines() -> DeviceProperties {
        let mut device = bare();
        device.features.copy_async = true;
        device.features.tma.insert(Tma::Base);
        device.features.types.opaque.insert(OpaqueType::Barrier);
        device
    }

    #[test]
    fn the_units_issue_their_own_lines_and_tma_the_whole_stage() {
        for delivery in [
            Delivery::SyncPerUnit,
            Delivery::AsyncPerUnit,
            Delivery::Procedural,
        ] {
            assert_eq!(delivery.scope(), ComputeScope::Unit, "{delivery:?}");
        }
        assert_eq!(Delivery::Tma.scope(), ComputeScope::Cube);
        assert_eq!(Delivery::Tma.order(), None);
    }

    /// The copy engine bypasses the cache, so its units read the source as it lies; the units'
    /// own loads go through it and write the stage as it lies.
    #[test]
    fn the_copy_engine_walks_the_source_and_the_units_the_destination() {
        assert_eq!(Delivery::AsyncPerUnit.order(), Some(LineOrder::Source));
        assert_eq!(Delivery::SyncPerUnit.order(), Some(LineOrder::Destination));
        assert_eq!(Delivery::Procedural.order(), Some(LineOrder::Destination));
    }

    #[test]
    fn only_the_units_land_their_lines_when_the_fill_returns() {
        assert_eq!(Delivery::SyncPerUnit.landing(), Landing::OnReturn);
        assert_eq!(Delivery::Procedural.landing(), Landing::OnReturn);
        assert_eq!(Delivery::AsyncPerUnit.landing(), Landing::Arrivals);
        assert_eq!(Delivery::Tma.landing(), Landing::Bytes);
        assert_eq!(Delivery::SyncPerUnit.rendezvous(), Rendezvous::Cube);
        assert_eq!(Delivery::AsyncPerUnit.rendezvous(), Rendezvous::Barrier);
    }

    #[test]
    fn an_engine_runs_where_the_device_has_it_and_a_barrier() {
        let (bare, engines) = (bare(), engines());
        for delivery in [Delivery::SyncPerUnit, Delivery::Procedural] {
            assert_eq!(delivery.runs_on(&bare), Ok(()));
        }
        for delivery in [Delivery::AsyncPerUnit, Delivery::Tma] {
            assert_eq!(
                delivery.runs_on(&bare),
                Err(Refusal::DeliveryNotOnDevice(delivery))
            );
            assert_eq!(delivery.runs_on(&engines), Ok(()));
        }
        let mut no_barrier = engines;
        no_barrier
            .features
            .types
            .opaque
            .remove(&OpaqueType::Barrier);
        assert_eq!(
            Delivery::AsyncPerUnit.runs_on(&no_barrier),
            Err(Refusal::DeliveryNotOnDevice(Delivery::AsyncPerUnit))
        );
    }

    #[test]
    fn a_register_prefetch_holds_only_what_the_units_load_themselves() {
        assert_eq!(Delivery::SyncPerUnit.holds(Prefetch::InRegisters), Ok(()));
        for delivery in [Delivery::AsyncPerUnit, Delivery::Tma] {
            assert_eq!(delivery.holds(Prefetch::InSlots), Ok(()));
            assert_eq!(
                delivery.holds(Prefetch::InRegisters),
                Err(Refusal::PrefetchPastDelivery(delivery))
            );
        }
    }

    #[test]
    fn the_copy_engine_moves_lines_of_4_8_or_16_bytes() {
        for bits in [32, 64, 128] {
            assert_eq!(Delivery::AsyncPerUnit.moves_lines(bits), Ok(()));
        }
        for bits in [8, 16, 256] {
            assert_eq!(
                Delivery::AsyncPerUnit.moves_lines(bits),
                Err(Refusal::LineNotCopied { bits })
            );
            assert_eq!(Delivery::SyncPerUnit.moves_lines(bits), Ok(()));
        }
    }
}
