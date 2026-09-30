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

#[cfg(all(test, std_io))]
mod tests {
    use alloc::string::String;

    use super::*;

    fn counter_that_failed_with(message: &str) -> UtilizationSource {
        UtilizationSource::Unavailable(UtilizationUnavailable::QueryFailed(String::from(message)))
    }

    #[test]
    fn asking_twice_for_one_counter_hands_back_the_one_opened_first() {
        let first = CardCounters::opened_for(counter_that_failed_with("asked for twice"));
        let second = CardCounters::opened_for(counter_that_failed_with("asked for twice"));

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn two_counters_are_opened_and_kept_apart() {
        let first = CardCounters::opened_for(counter_that_failed_with("the first of two"));
        let second = CardCounters::opened_for(counter_that_failed_with("the second of two"));

        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(
            first.read(),
            Err(UtilizationUnavailable::QueryFailed(String::from(
                "the first of two"
            )))
        );
        assert_eq!(
            second.read(),
            Err(UtilizationUnavailable::QueryFailed(String::from(
                "the second of two"
            )))
        );
    }
}
