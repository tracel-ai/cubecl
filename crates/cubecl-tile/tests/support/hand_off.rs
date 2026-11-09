use super::ValidationResult;
use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, Type, UIntKind},
};

/// Whether this device hands one cube's writes to another within a dispatch and stores and adds
/// `u32`s atomically, what a kernel needs where cubes publish to each other; reported rather than
/// silently passed.
pub fn hands_off_between_cubes(client: &cubecl::client::Client) -> bool {
    let hands_off = client.properties().features.device_memory_scope;
    let usage = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::UInt(UIntKind::U32)));
    let counts = usage.contains(AtomicUsage::LoadStore) && usage.contains(AtomicUsage::Add);
    if !(hands_off && counts) {
        ValidationResult::Skipped(
            "device has no device-scope storage sync or no u32 atomic store and add".to_string(),
        )
        .enforce();
    }
    hands_off && counts
}
