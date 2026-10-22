use cubecl::prelude::Client;

/// Whether a kernel's result matched its reference.
#[derive(Debug)]
pub enum ValidationResult {
    Pass,
    Fail(String),
    /// The test does not apply to this runtime.
    Skipped(String),
}

impl ValidationResult {
    /// Fails the test on a mismatch; a skip passes, saying why.
    #[track_caller]
    pub fn enforce(self) {
        match self {
            ValidationResult::Pass => {}
            ValidationResult::Fail(reason) => panic!("Test failed: {reason}"),
            ValidationResult::Skipped(reason) => eprintln!("Skipped: {reason}"),
        }
    }
}

/// A kernel whose levels cut down to units needs a plane wide enough to hold them: a unit level
/// must partition the plane exactly, so a plane of one unit (what the CPU runtime always reports)
/// leaves such a partitioning with nothing to cut. Records a `Skipped` outcome and returns `true`
/// when `client`'s plane is narrower than `units`, so the caller can bail out of the test.
#[track_caller]
pub fn skip_unless_plane_holds(client: &Client, units: u32) -> bool {
    let plane = client.properties().hardware.plane_size_max;
    if plane < units {
        ValidationResult::Skipped(format!(
            "needs a plane of at least {units} units; this runtime reports {plane}"
        ))
        .enforce();
        return true;
    }
    false
}
