use alloc::vec::Vec;
use core::cell::OnceCell;
use cubecl_environment::sync::Mutex;
use cubecl_runtime::{
    client::{Client, ClientGroup},
    execution::{StreamMode, StreamModeOverride},
    logging::TransferRoute,
    server::PeerAccess,
    throughput::{
        DEFAULT_WORKING_SET_BYTES, KernelConfig, LinkKey, LinkRoute, MemoryCurve, MemoryPoint,
        ThroughputBenchmarker, ThroughputCache, ThroughputError, ThroughputValue, Traffic,
        working_set_sweep,
    },
};

use crate::throughput::{
    ShapeSweep, link_collective, link_peer, link_peer::MAX_BYTES, link_transfer,
};

/// One link probe at a time in the process. A probe holds several devices, which no device's
/// exclusive hold can do: it runs on that device's thread, which the probe then calls into.
static LINK_PROBES: Mutex<()> = Mutex::new(());

/// Measures `key` between `members`, in order, cached for them.
///
/// A pair's route reads the first member as the bytes' source and the second as where they go; a
/// collective runs over every member.
///
/// Native only: a probe blocks on the devices, which the browser can't do.
///
/// # Errors
///
/// [`Unsupported`](ThroughputError::Unsupported) where the members cannot move bytes that way,
/// a member skips its communication, or the route needs a count of members it was not given;
/// otherwise as [`measure_peak_throughput`](super::measure_peak_throughput).
pub fn measure_link_throughput(
    members: &[Client],
    key: LinkKey,
) -> Result<ThroughputValue, ThroughputError> {
    LinkProber::new(members).measure(key)
}

/// The route `Client::to_client` takes from `source` to `destination`: the device transport
/// between two devices of one runtime that has one, the host otherwise.
pub fn transfer_route(source: &Client, destination: &Client) -> TransferRoute {
    let same_runtime = source.service_id().service == destination.service_id().service;
    match source.has_device_transport() && same_runtime {
        true => TransferRoute::Device,
        false => TransferRoute::Host,
    }
}

/// Measures `route` between `members` across a range of sizes, from a few kilobytes up to the
/// default working set or as much as every member will allocate.
///
/// Returns an empty curve on WASM, or where the members cannot move bytes that way.
pub fn measure_link_curve(members: &[Client], route: LinkRoute) -> MemoryCurve {
    let prober = LinkProber::new(members);
    let points = working_set_sweep(prober.size_cap(route))
        .into_iter()
        .filter_map(|bytes| {
            let value = prober.measure(LinkKey { route, bytes }).ok()?;
            Some(MemoryPoint { bytes, value })
        })
        .collect::<Vec<_>>();

    MemoryCurve::new(Traffic::Link(route), points)
}

/// Measures the routes between one set of members, holding the group a collective needs across
/// the points of a curve, since setting one up costs more than a small collective.
struct LinkProber<'a> {
    members: &'a [Client],
    group: OnceCell<Option<ClientGroup>>,
}

impl<'a> LinkProber<'a> {
    fn new(members: &'a [Client]) -> Self {
        Self {
            members,
            group: OnceCell::new(),
        }
    }

    fn measure(&self, key: LinkKey) -> Result<ThroughputValue, ThroughputError> {
        #[cfg(target_family = "wasm")]
        {
            let _ = key;
            Err(ThroughputError::Unsupported)
        }

        #[cfg(not(target_family = "wasm"))]
        {
            // Before the cache: a route the members no longer take would answer the rate of one
            // they took before.
            if !self.supports(key.route) || key.bytes > MAX_BYTES {
                return Err(ThroughputError::Unsupported);
            }
            let Some(first) = self.members.first() else {
                return Err(ThroughputError::Unsupported);
            };
            let properties = self
                .members
                .iter()
                .map(Client::properties)
                .collect::<Vec<_>>();
            let benchmarker = ThroughputCache::get_for_link(first.name(), &properties)
                .map(ThroughputBenchmarker::new);
            if let Some(value) = benchmarker.as_ref().and_then(|cached| cached.cached(key)) {
                return Ok(value);
            }

            let _probing = LINK_PROBES.lock();
            match benchmarker {
                Some(mut benchmarker) => benchmarker.measure(key, || self.probe(key)),
                None => self.probe(key),
            }
        }
    }

    fn probe(&self, key: LinkKey) -> Result<ThroughputValue, ThroughputError> {
        // Each member's stream executes whatever the process mode drops.
        let _executing = self
            .members
            .iter()
            .map(|client| StreamModeOverride::new(StreamMode::Execute, client))
            .collect::<Vec<_>>();
        for client in self.members {
            client.compile_queued();
        }

        ShapeSweep::new(alloc::vec![key.bytes])
            .fastest(|bytes| self.build(key.route, bytes))
            .map(|(value, _)| value)
    }

    fn build(&self, route: LinkRoute, bytes: u64) -> Result<KernelConfig, ThroughputError> {
        let [source, destination, ..] = self.members else {
            return Err(ThroughputError::Unsupported);
        };
        match route {
            LinkRoute::Transfer(_) => link_transfer::build_kernel(source, destination, bytes),
            LinkRoute::PeerLoad => link_peer::build_load(source, destination, bytes),
            LinkRoute::PeerStore => link_peer::build_store(source, destination, bytes),
            LinkRoute::Collective(kind) => self
                .group()
                .ok_or(ThroughputError::Unsupported)
                .and_then(|group| link_collective::build_kernel(group, kind, bytes)),
        }
    }

    /// Whether the members move bytes over `route` now.
    fn supports(&self, route: LinkRoute) -> bool {
        if self.members.iter().any(Client::skips_communication) {
            return false;
        }
        match (route, self.members) {
            (LinkRoute::Transfer(taken), [source, destination]) => {
                transfer_route(source, destination) == taken
            }
            (LinkRoute::PeerLoad, [source, destination]) => {
                destination.peer_access(source) >= PeerAccess::Memory
            }
            (LinkRoute::PeerStore, [source, destination]) => {
                source.peer_access(destination) >= PeerAccess::Memory
            }
            (LinkRoute::Collective(_), [_, _, ..]) => self.group().is_some(),
            _ => false,
        }
    }

    /// The members as a group, or `None` where they cannot be one: no device transport, mixed
    /// runtimes, or devices out of order. Building one sets up no communicator.
    fn group(&self) -> Option<&ClientGroup> {
        self.group
            .get_or_init(|| ClientGroup::new(self.members.to_vec()).ok())
            .as_ref()
    }

    /// The largest size a curve of `route` measures: the default working set, or less where a
    /// member cannot allocate the buffers it holds at that size.
    fn size_cap(&self, route: LinkRoute) -> u64 {
        let buffers = match route {
            LinkRoute::Collective(_) => 2,
            LinkRoute::Transfer(_) | LinkRoute::PeerLoad | LinkRoute::PeerStore => 1,
        };
        self.members
            .iter()
            .map(|client| client.properties().memory.max_page_size / buffers)
            .fold(DEFAULT_WORKING_SET_BYTES, u64::min)
    }
}

impl core::fmt::Debug for LinkProber<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LinkProber")
            .field("members", &self.members.len())
            .finish()
    }
}
