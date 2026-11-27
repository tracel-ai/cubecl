/// How a tune holds the device it measures while it runs its candidates.
///
/// # Guarantees
///
/// - [`Exclusive`](Self::Exclusive): nothing else is scheduled on the device while a batch or a
///   sample runs, and a candidate's panic is contained by the device's own thread.
/// - [`Shared`](Self::Shared): the device is never held, and a candidate's panic is contained on
///   the calling thread with `std`; without `std`, it unwinds through the call that tuned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuneDeviceAccess {
    /// The device is held to itself for every batch and every sample, nothing else scheduled
    /// against it: the candidates run on the device's own thread.
    #[default]
    Exclusive,
    /// The device is never held: the candidates run on the calling thread, each sample timed by a
    /// window opened and closed on the calling stream. Work a candidate hands to another thread,
    /// such as a lazy queue drained there, reaches the device and lands on that stream; work of
    /// other streams may run beside a sample and slow it.
    Shared,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tune_holds_its_device_unless_told_otherwise() {
        assert_eq!(TuneDeviceAccess::default(), TuneDeviceAccess::Exclusive);
    }
}
