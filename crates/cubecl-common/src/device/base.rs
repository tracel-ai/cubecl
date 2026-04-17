use crate::stub::Arc;
use core::{any::Any, cmp::Ordering};

use derive_new::new;

/// Uniquely identifies a device within the Burn/CubeCL stack.
///
/// A `DeviceId` is packed into 32 bits, split across three fields:
///
/// ```text
///  31          24 23          16 15                           0
/// ┌──────────────┬──────────────┬──────────────────────────────┐
/// │     role     │     kind     │           index_id           │
/// │    (u8)      │    (u8)      │            (u16)             │
/// └──────────────┴──────────────┴──────────────────────────────┘
/// ```
///
/// The `role` describes which execution layer owns the device (autodiff,
/// fusion, or the underlying runtime), the `kind` describes the hardware
/// category, and the `index_id` distinguishes between multiple devices of
/// the same role and kind (e.g. GPU 0 vs GPU 1).
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The execution layer the device belongs to.
    ///
    /// Burn stacks execution layers on top of a runtime (autodiff wraps
    /// fusion wraps runtime, for example), and each layer exposes its own
    /// view of the underlying device. The role disambiguates these views
    /// so that, for instance, an autodiff-wrapped GPU 0 is not confused
    /// with the raw runtime GPU 0.
    pub role: DeviceRole,
    /// The hardware category of the device.
    pub kind: DeviceKind,
    /// The index of the device among devices sharing the same role and
    /// kind.
    ///
    /// For example, on a machine with two discrete GPUs this is `0` for
    /// the first and `1` for the second. Always `0` when only a single
    /// device of that role and kind exists.
    pub index_id: u16,
}

/// The execution layer a device belongs to.
///
/// Burn composes functionality by stacking layers over a base runtime.
/// A device's role indicates which layer in that stack currently owns
/// it, which matters because each layer tracks its own state (autograd
/// graphs, fused operation streams, etc.) keyed by device.
#[repr(u8)]
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum DeviceRole {
    /// The device is managed by the autodiff layer, which records
    /// operations to build a computation graph for backward passes.
    Autodiff = 0,
    /// The device is managed by the fusion layer, which batches and
    /// fuses compatible operations before dispatching them to the
    /// runtime.
    Fusion = 1,
    /// The device is managed directly by the underlying runtime with no
    /// additional wrapping layer.
    Runtime = 2,
}

/// The hardware category of a device.
///
/// This describes the physical nature of the compute unit backing the
/// device and is useful for scheduling decisions, memory placement, and
/// reporting.
#[repr(u8)]
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum DeviceKind {
    /// A dedicated GPU with its own memory, typically connected over
    /// PCIe. Offers the highest throughput for large tensor workloads.
    DiscreteGpu = 0,
    /// A GPU that shares memory with the host CPU, commonly found in
    /// laptops, mobile SoCs, and APUs. Lower peak throughput than a
    /// discrete GPU but avoids host-device memory transfers.
    IntegratedGpu = 1,
    /// A CPU used as a compute device, either as the primary target or
    /// as a fallback when no GPU is available.
    Cpu = 2,
    /// A virtualized or software-emulated GPU, such as those exposed by
    /// hypervisors, remote rendering services, or software rasterizers.
    /// Performance characteristics are backend-dependent.
    VirtualGpu = 3,
}

/// Device trait for all cubecl devices.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync + 'static {
    /// Create a device from its [id](DeviceId).
    fn from_id(device_id: DeviceId) -> Self;
    /// Retrieve the [device id](DeviceId) from the device.
    fn to_id(&self) -> DeviceId;
}

impl core::fmt::Display for DeviceRole {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            DeviceRole::Autodiff => "autodiff",
            DeviceRole::Fusion => "fusion",
            DeviceRole::Runtime => "runtime",
        };
        f.write_str(name)
    }
}

impl core::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            DeviceKind::DiscreteGpu => "discrete-gpu",
            DeviceKind::IntegratedGpu => "integrated-gpu",
            DeviceKind::Cpu => "cpu",
            DeviceKind::VirtualGpu => "virtual-gpu",
        };
        f.write_str(name)
    }
}

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DeviceId(role={}, kind={}, index={})",
            self.role, self.kind, self.index_id
        )
    }
}

impl Ord for DeviceId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.role
            .cmp(&other.role)
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.index_id.cmp(&other.index_id))
    }
}

impl PartialOrd for DeviceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An pointer to a service's server utilities.
pub type ServerUtilitiesHandle = Arc<dyn Any + Send + Sync>;

/// Represent a service that runs on a device.
pub trait DeviceService: Send + 'static {
    /// Initializes the service. It is only called once per device.
    fn init(device_id: DeviceId) -> Self;
    /// Get the service utilities.
    fn utilities(&self) -> ServerUtilitiesHandle;
}
