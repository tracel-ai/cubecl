use super::HostMemoryApi;
use crate::driver::DriverError;
use alloc::vec::Vec;
use cubecl_environment::bytes::Bytes;
use cubecl_environment::sync::Mutex;

/// Pins host memory on its first import and unpins it on its last release, counting the imports
/// of each memory between the two.
pub struct HostPinner {
    /// Keyed by the memory's address and length: the imports the devices hold. A count of zero is
    /// memory the driver failed to unpin, which it still holds. A handful of tables at most.
    pins: Mutex<Vec<((usize, usize), usize)>>,
}

impl HostPinner {
    /// No memory pinned.
    pub const fn new() -> Self {
        Self {
            pins: Mutex::new(Vec::new()),
        }
    }

    /// Counts an import of `memory`, pinning it when the driver holds no pin of it.
    pub fn pin<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        let key = key(memory);
        let mut pins = self.pins.lock();
        if let Some((_, imports)) = pins.iter_mut().find(|(pinned, _)| *pinned == key) {
            *imports += 1;
            return Ok(());
        }
        // SAFETY: the import holds `memory` until its last release unregisters it.
        unsafe { A::host_register(memory)? };
        pins.push((key, 1));
        Ok(())
    }

    /// Counts a release of `memory`, unpinning it on the last. Memory the driver fails to unpin
    /// stays counted as pinned, since the driver still holds it.
    pub fn unpin<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        let key = key(memory);
        let mut pins = self.pins.lock();
        let at = pins
            .iter()
            .position(|(pinned, imports)| *pinned == key && *imports > 0)
            .expect("a released import should have been pinned");
        pins[at].1 -= 1;
        if pins[at].1 > 0 {
            return Ok(());
        }
        // SAFETY: it pinned on its first import, and its last release runs once no device reads it.
        unsafe { A::host_unregister(memory)? };
        pins.swap_remove(at);
        Ok(())
    }
}

fn key(memory: &Bytes) -> (usize, usize) {
    (memory.as_ptr() as usize, memory.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::HOST_MEMORY_ALIGNMENT_BYTES;
    use crate::storage::host_memory_fake::{Fake, devices};

    fn memory() -> Bytes {
        Bytes::zeroed_aligned(HOST_MEMORY_ALIGNMENT_BYTES, HOST_MEMORY_ALIGNMENT_BYTES).unwrap()
    }

    #[test]
    fn memory_pinned_again_is_counted_not_pinned() {
        let (driver, ..) = devices();
        let pinner = HostPinner::new();
        let memory = memory();
        pinner.pin::<Fake>(&memory).unwrap();
        pinner.pin::<Fake>(&memory).unwrap();
        assert_eq!(driver.pins.get(), 1);
    }

    #[test]
    fn only_the_last_unpin_unregisters() {
        let (driver, ..) = devices();
        let pinner = HostPinner::new();
        let memory = memory();
        pinner.pin::<Fake>(&memory).unwrap();
        pinner.pin::<Fake>(&memory).unwrap();
        pinner.unpin::<Fake>(&memory).unwrap();
        assert_eq!(driver.unpins.get(), 0);
        pinner.unpin::<Fake>(&memory).unwrap();
        assert_eq!(driver.unpins.get(), 1);
    }

    #[test]
    fn memory_overlapping_a_pin_pins_on_its_own() {
        let (driver, ..) = devices();
        let pinner = HostPinner::new();
        let whole =
            Bytes::zeroed_aligned(2 * HOST_MEMORY_ALIGNMENT_BYTES, HOST_MEMORY_ALIGNMENT_BYTES)
                .unwrap()
                .shared();
        let prefix = whole.view(0, HOST_MEMORY_ALIGNMENT_BYTES).unwrap();
        pinner.pin::<Fake>(&prefix).unwrap();
        pinner.pin::<Fake>(&whole).unwrap();
        assert_eq!(driver.pins.get(), 2);
    }

    #[test]
    fn a_refused_pin_counts_nothing() {
        let (driver, ..) = devices();
        let pinner = HostPinner::new();
        let memory = memory();
        driver.refuse_pin.set(true);
        assert!(pinner.pin::<Fake>(&memory).is_err());
        driver.refuse_pin.set(false);
        pinner.pin::<Fake>(&memory).unwrap();
        assert_eq!(driver.pins.get(), 1);
    }

    #[test]
    fn memory_that_failed_to_unpin_pins_again_without_the_driver() {
        let (driver, ..) = devices();
        let pinner = HostPinner::new();
        let memory = memory();
        pinner.pin::<Fake>(&memory).unwrap();
        driver.refuse_unpin.set(true);
        assert!(pinner.unpin::<Fake>(&memory).is_err());
        driver.refuse_unpin.set(false);
        pinner.pin::<Fake>(&memory).unwrap();
        assert_eq!(driver.pins.get(), 1);
        pinner.unpin::<Fake>(&memory).unwrap();
        assert_eq!(driver.unpins.get(), 1);
    }
}
