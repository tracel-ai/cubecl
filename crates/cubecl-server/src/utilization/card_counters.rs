#[cfg(std_io)]
use alloc::{sync::Arc, vec::Vec};
#[cfg(std_io)]
use std::sync::{Mutex, PoisonError};

use cubecl_runtime::client::Client;

#[cfg(std_io)]
use super::{opened::OpenedCounter, source::UtilizationSource};
use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// The counters of the cards the GPU runtimes drive, each opened the first time its card is asked
/// and kept for the life of the process: opening is the costly half, loading NVML or opening a
/// query.
pub struct CardCounters;

#[cfg(std_io)]
static OPENED_COUNTERS: Mutex<Vec<(UtilizationSource, Arc<OpenedCounter>)>> =
    Mutex::new(Vec::new());

impl CardCounters {
    /// How busy the card behind `client`'s device is. The card and the platform decide which
    /// counter answers, never the runtime: an NVIDIA card reads the same through CUDA as through
    /// wgpu.
    #[cfg(std_io)]
    pub fn read_card_behind(client: &Client) -> Result<DeviceUtilization, UtilizationUnavailable> {
        let card = client.properties().identity.physical.as_ref();
        Self::opened_for(UtilizationSource::of_card(card)).read()
    }

    /// How busy the card behind `client`'s device is: never known off a desktop platform, where no
    /// counter is read.
    #[cfg(not(std_io))]
    pub fn read_card_behind(client: &Client) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Err(match client.properties().identity.physical.as_ref() {
            Some(card) => UtilizationUnavailable::NoCounterForCard(card.vendor),
            None => UtilizationUnavailable::NoCard,
        })
    }

    /// Keyed by the counter rather than the device, so the devices of every runtime on one card
    /// share it.
    #[cfg(std_io)]
    fn opened_for(source: UtilizationSource) -> Arc<OpenedCounter> {
        let mut opened_counters = OPENED_COUNTERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((_, opened)) = opened_counters
            .iter()
            .find(|(opened_source, _)| *opened_source == source)
        {
            return opened.clone();
        }
        let opened = Arc::new(OpenedCounter::open(source.clone()));
        opened_counters.push((source, opened.clone()));
        opened
    }
}
