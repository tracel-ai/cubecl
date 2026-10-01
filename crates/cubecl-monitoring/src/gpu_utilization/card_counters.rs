#[cfg(card_counter)]
use alloc::{sync::Arc, vec::Vec};
#[cfg(card_counter)]
use std::sync::{Mutex, PoisonError};

use cubecl_ir::PhysicalDevice;

#[cfg(card_counter)]
use super::{opened::OpenedCounter, source::UtilizationSource};
use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// The counters of the cards the GPU runtimes drive, each opened the first time its card is asked
/// and kept for the life of the process: opening is the costly half, loading NVML or opening a
/// query.
pub struct CardCounters;

#[cfg(card_counter)]
static OPENED_COUNTERS: Mutex<Vec<(UtilizationSource, Arc<OpenedCounter>)>> =
    Mutex::new(Vec::new());

impl CardCounters {
    /// How busy `card` is, the card a runtime reports its device runs on. The card and the
    /// platform decide which counter answers, never the runtime: an NVIDIA card reads the same
    /// through CUDA as through wgpu.
    #[cfg(card_counter)]
    pub fn read(
        card: Option<&PhysicalDevice>,
    ) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Self::opened_for(UtilizationSource::of_card(card)).read()
    }

    /// How busy `card` is: never known to a build that compiles no counter for this platform.
    #[cfg(not(card_counter))]
    pub fn read(
        card: Option<&PhysicalDevice>,
    ) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Err(match card {
            Some(card) => UtilizationUnavailable::NoCounterForCard(card.vendor),
            None => UtilizationUnavailable::NoCard,
        })
    }

    /// Keyed by the counter rather than the device, so the devices of every runtime on one card
    /// share it.
    #[cfg(card_counter)]
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

#[cfg(all(test, card_counter))]
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
