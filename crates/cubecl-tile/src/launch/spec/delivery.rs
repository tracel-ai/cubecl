//! How an operand's bytes reach a stage ([`Delivery`]): the engine that moves them. Each engine
//! states what it is ([`Engine`]): who issues one move ([`Issuer`]), the route a line takes
//! ([`Route`]), and what the device must have; when what it moved has landed ([`Landing`]) follows
//! from those. Each engine is one file under `stage/engine/`, which states its [`Engine`] and
//! makes its move, and [`Delivery::engine`] is the one place a delivery is matched on for what
//! it states.

use cubecl::ir::{DeviceProperties, OpaqueType};

use crate::stage::{ASYNC_PER_UNIT, Prefetch, SYNC_PER_UNIT, TMA};
use crate::{Refusal, Rendezvous, Space};

/// The engine that moves an operand's bytes into a stage filled from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    /// Every unit loads its own lines and stores them, through its registers: landed when the fill
    /// returns, and a line can be decoded on its way.
    #[default]
    SyncPerUnit,
    /// Every unit hands its own lines to the copy engine (`cp.async`), memory to memory: they land
    /// after the fill returns, as they lie.
    AsyncPerUnit,
    /// One unit hands the stage's box to the TMA engine through a tensor map
    /// (`cp.async.bulk.tensor`): it lands after the fill returns, counted in bytes.
    Tma,
}

/// What an engine states about itself: everything a stage, its slot and a routine choosing a
/// delivery ask of it. One per engine, in the engine's own file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Engine {
    /// Who issues one move.
    pub(crate) issuer: Issuer,
    /// The route a line takes.
    pub(crate) route: Route,
    /// Whether a device has the engine. A landing after the fill returns also needs the barrier
    /// it lands on, which [`Delivery::runs_on`] asks for every engine.
    pub(crate) present: fn(&DeviceProperties) -> bool,
    /// The line lengths it moves whole, in bytes; `None` for any.
    pub(crate) line_bytes: Option<&'static [usize]>,
    /// The most cells a box it moves holds along one axis; `None` for no bound.
    pub(crate) box_edge: Option<usize>,
    /// The byte alignment a stage it lands in starts on; `0` for none.
    pub(crate) stage_alignment: usize,
}

/// Who issues one move of an engine: the granularity it moves at. Not which units share a fill's
/// moves, which the level the stage's copies deal their lines by says
/// ([`Level::copied_by`](crate::Level::copied_by)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Issuer {
    /// Each unit moves its own lines.
    Unit,
    /// One elected unit moves the whole stage of the cube that owns it.
    Cube,
}

/// The route a line takes from its source to its stage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Route {
    /// Through a unit's registers, where it can be decoded, padded, scanned or summed.
    Registers,
    /// Memory to memory, as it lies, without a unit touching it.
    MemoryToMemory,
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

impl Engine {
    /// When what this engine moved has landed: when the fill returns where the units wrote it
    /// themselves; after, on what the slot's barrier counts, where it went memory to memory: each
    /// unit's arrivals where every unit issued, the bytes the one issuer announced otherwise.
    pub(crate) fn landing(&self) -> Landing {
        match (self.route, self.issuer) {
            (Route::Registers, _) => Landing::OnReturn,
            (Route::MemoryToMemory, Issuer::Unit) => Landing::Arrivals,
            (Route::MemoryToMemory, Issuer::Cube) => Landing::Bytes,
        }
    }
}

impl Delivery {
    /// The byte alignment a stage a [`Tma`](Delivery::Tma) box lands in starts on: the span a
    /// 128-byte swizzle repeats over, which the engine keys off the address, so a stage's first
    /// row takes the first key as the fragment reading it does. A launch sizing its shared memory
    /// counts what aligning a stage to it can cost.
    pub const TMA_STAGE_ALIGNMENT: usize = TMA.stage_alignment;

    /// What this delivery's engine states.
    pub(crate) fn engine(self) -> &'static Engine {
        match self {
            Delivery::SyncPerUnit => &SYNC_PER_UNIT,
            Delivery::AsyncPerUnit => &ASYNC_PER_UNIT,
            Delivery::Tma => &TMA,
        }
    }

    /// Whether this delivery moves `stage` in one transfer, and why not where it cannot.
    pub fn moves(self, stage: &Space) -> Result<(), Refusal> {
        let Some(most) = self.engine().box_edge else {
            return Ok(());
        };
        match stage.extents().into_iter().find(|&(_, edge)| edge > most) {
            Some((axis, edge)) => Err(Refusal::BoxPastDescriptor { axis, edge, most }),
            None => Ok(()),
        }
    }

    /// Whether `device` has what this delivery runs on: its engine, and the barrier what it moves
    /// lands on where that is after the fill returns.
    pub fn runs_on(self, device: &DeviceProperties) -> Result<(), Refusal> {
        let barriers = match self.landing() {
            Landing::OnReturn => true,
            Landing::Arrivals | Landing::Bytes => {
                device.features.types.opaque.contains(&OpaqueType::Barrier)
            }
        };
        match (self.engine().present)(device) && barriers {
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

    /// Whether this delivery moves lines `bits` long whole.
    pub fn moves_lines(self, bits: usize) -> Result<(), Refusal> {
        match self.engine().line_bytes {
            Some(lengths) if !lengths.iter().any(|&bytes| bytes * 8 == bits) => {
                Err(Refusal::LineNotCopied { bits })
            }
            Some(_) | None => Ok(()),
        }
    }

    /// Who issues one move ([`Engine::issuer`]).
    pub(crate) fn issuer(self) -> Issuer {
        self.engine().issuer
    }

    /// The route a line takes ([`Engine::route`]).
    pub(crate) fn route(self) -> Route {
        self.engine().route
    }

    /// When what this delivery moved has landed in the stage ([`Engine::landing`]).
    pub(crate) fn landing(self) -> Landing {
        self.engine().landing()
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
    use cubecl::ir::features::Tma;
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
        for delivery in [Delivery::SyncPerUnit, Delivery::AsyncPerUnit] {
            assert_eq!(delivery.issuer(), Issuer::Unit, "{delivery:?}");
        }
        assert_eq!(Delivery::Tma.issuer(), Issuer::Cube);
    }

    /// A line through the registers lands with the fill; one moved memory to memory lands on
    /// each unit's arrivals where every unit issued, on the announced bytes where one did.
    #[test]
    fn the_landing_follows_from_the_path_and_the_issuer() {
        assert_eq!(Delivery::SyncPerUnit.landing(), Landing::OnReturn);
        assert_eq!(Delivery::AsyncPerUnit.landing(), Landing::Arrivals);
        assert_eq!(Delivery::Tma.landing(), Landing::Bytes);
        assert_eq!(Delivery::SyncPerUnit.rendezvous(), Rendezvous::Cube);
        assert_eq!(Delivery::AsyncPerUnit.rendezvous(), Rendezvous::Barrier);
    }

    #[test]
    fn an_engine_runs_where_the_device_has_it_and_a_barrier() {
        let (bare, engines) = (bare(), engines());
        assert_eq!(Delivery::SyncPerUnit.runs_on(&bare), Ok(()));
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
