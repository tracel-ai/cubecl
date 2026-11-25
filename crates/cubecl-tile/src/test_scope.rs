//! The scope the host-side tests of this crate expand into ([`test_scope`]).

use cubecl::ir::{AddressType, Scope, settings};

/// A root scope of one unit, checked, addressed in `u32`.
pub(crate) fn test_scope() -> Scope {
    Scope::root(settings::KernelSettings::new(
        settings::Dim3::new_single(),
        settings::ExecutionMode::Checked,
        AddressType::U32,
    ))
}
