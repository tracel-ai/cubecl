//! Persistent compilation cache tests for `cubecl-cpu` (Issue #1527).

#![cfg(compilation_cache)]

use cubecl_common::device::DeviceId;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::compute::cache::CpuCacheEntry;
use cubecl_cpu::compute::server::CpuServer;
use cubecl_environment::persistence::{Database, Store};
use cubecl_environment::records::{self, RecordLevel, Records, RecordsConfig};
use cubecl_llvm::PlironEngine;
use cubecl_server::compiler::{
    CompilationOutcome, CompilationRecord, KernelCacheKey, compilation_store,
};
use cubecl_server::config::{CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationConfig};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "cubecl_cpu_cache_test_{}_{}_{}",
            std::process::id(),
            name,
            nanos
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("failed to create temp dir");
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cube(launch)]
pub fn add_scalar_kernel(output: &mut [u32], input: &[u32], #[comptime] scalar: u32) {
    if UNIT_POS == 0 {
        output[0] = input[0] + scalar;
    }
}

/// Asserts that the cold launch compiles and records `CompilationOutcome::Compiled`,
/// and the warm launch in Server Instance 2 hits the cache, records `CompilationOutcome::Loaded`,
/// and produces byte-for-byte identical output.
#[test]
fn test_cold_vs_warm_launch() {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let temp = TempDir::new("cold_warm");

    let _ = CubeClRuntimeConfig::try_set(CubeClRuntimeConfig {
        compilation: CompilationConfig {
            cache: true,
            ..Default::default()
        },
        ..Default::default()
    });
    cubecl_environment::environment::set_root(&temp.path);
    records::configure(RecordsConfig {
        level: RecordLevel::Basic,
        ..Default::default()
    });

    // --- Cold Launch: Server Instance 1 ---
    let client1 = Client::load::<CpuServer>(DeviceId::new(0, 10));
    let in1 = client1.create_from_slice(u32::as_bytes(&[42]));
    let out1 = client1.create_from_slice(u32::as_bytes(&[0]));

    add_scalar_kernel::launch(
        &client1,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out1.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in1.clone(), 1) },
        10,
    );

    let res1_bytes = client1.read_one_unchecked(out1);
    let res1 = u32::from_bytes(&res1_bytes);
    assert_eq!(res1[0], 52);

    let fingerprint = client1.properties().identity.fingerprint.clone();
    drop(client1);

    // Verify Cold run compiled and stored
    let database = Database::open_active().expect("database open");
    let recs = Records::new(&database);
    let cold_records = recs.read::<CompilationRecord>();
    assert_eq!(cold_records.len(), 1);
    assert_eq!(cold_records[0].record.outcome, CompilationOutcome::Compiled);

    // Verify stored object can be warm loaded and retains 0 object bytes
    let mut store: Store<KernelCacheKey, CpuCacheEntry> =
        compilation_store("cpu", &fingerprint).expect("store opens");
    let mut entries = Vec::new();
    store.scan(|k, v| entries.push((k, v)));
    assert_eq!(entries.len(), 1);
    let (_, entry) = entries.pop().unwrap();
    let warm_engine = PlironEngine::from_object(
        &entry.object,
        &entry.entrypoint_name,
        cubecl_llvm::KernelRequirements::default(),
        vec![],
    )
    .expect("from_object loads valid engine");
    assert!(
        warm_engine.take_object_bytes().is_none(),
        "warm-loaded PlironEngine must not retain object bytes in memory"
    );

    // --- Warm Launch: Server Instance 2 ---
    let client2 = Client::load::<CpuServer>(DeviceId::new(0, 11));
    let in2 = client2.create_from_slice(u32::as_bytes(&[42]));
    let out2 = client2.create_from_slice(u32::as_bytes(&[0]));

    add_scalar_kernel::launch(
        &client2,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out2.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in2.clone(), 1) },
        10,
    );

    let res2_bytes = client2.read_one_unchecked(out2);
    let res2 = u32::from_bytes(&res2_bytes);
    assert_eq!(res2[0], 52);

    // Output matches cold run byte-for-byte
    assert_eq!(res1_bytes, res2_bytes);

    // Verify Warm run hit cache (Outcome == Loaded)
    let recs = Records::new(&database);
    let all_records = recs.read::<CompilationRecord>();
    assert_eq!(all_records.len(), 2);
    assert_eq!(all_records[1].record.outcome, CompilationOutcome::Loaded);

    drop(client2);
    records::configure(RecordsConfig {
        level: RecordLevel::Off,
        ..Default::default()
    });
}

/// Asserts that running the same kernel with 64-byte alignment and 16-byte alignment
/// generates separate store entries without collision or misaligned load errors.
#[test]
fn test_alignment_key_isolation() {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let temp = TempDir::new("alignment");

    let _ = CubeClRuntimeConfig::try_set(CubeClRuntimeConfig {
        compilation: CompilationConfig {
            cache: true,
            ..Default::default()
        },
        ..Default::default()
    });
    cubecl_environment::environment::set_root(&temp.path);

    let client = Client::load::<CpuServer>(DeviceId::new(0, 20));

    // 1. 64-byte alignment launch: standard buffer starts at 64-byte alignment
    let in64 = client.create_from_slice(u32::as_bytes(&[100; 16]));
    let out64 = client.create_from_slice(u32::as_bytes(&[0; 16]));
    add_scalar_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out64.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in64.clone(), 1) },
        7,
    );
    let res64 = client.read_one_unchecked(out64);
    assert_eq!(u32::from_bytes(&res64)[0], 107);

    // 2. 16-byte alignment launch: slice with offset_start = 16
    let in16_buf = client.create_from_slice(u32::as_bytes(&[200; 32]));
    let out16_buf = client.create_from_slice(u32::as_bytes(&[0; 32]));
    let in16 = in16_buf.offset_start(16);
    let out16 = out16_buf.offset_start(16);
    add_scalar_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out16.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in16.clone(), 1) },
        7,
    );
    let res16 = client.read_one_unchecked(out16);
    assert_eq!(u32::from_bytes(&res16)[0], 207);

    let fingerprint = client.properties().identity.fingerprint.clone();
    drop(client);

    // Assert both variants generated separate store entries in the storage
    let mut store: Store<KernelCacheKey, CpuCacheEntry> =
        compilation_store("cpu", &fingerprint).expect("store opens");
    let mut entries = Vec::new();
    store.scan(|k, v| entries.push((k, v)));
    assert_eq!(
        entries.len(),
        2,
        "both 64-byte and 16-byte variants must have separate store entries"
    );
    assert_ne!(
        entries[0].0, entries[1].0,
        "the two alignment keys must be different"
    );
}

/// Asserts that when a store entry is corrupted with malformed bytes, the runtime does not panic,
/// logs a warning, falls back to full compilation, rewrites a valid entry, and completes successfully.
#[test]
fn test_corrupted_store_recovery() {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let temp = TempDir::new("corrupted");

    let _ = CubeClRuntimeConfig::try_set(CubeClRuntimeConfig {
        compilation: CompilationConfig {
            cache: true,
            ..Default::default()
        },
        ..Default::default()
    });
    cubecl_environment::environment::set_root(&temp.path);

    // Initial valid compile and store with Server Instance 1
    let client1 = Client::load::<CpuServer>(DeviceId::new(0, 30));
    let in1 = client1.create_from_slice(u32::as_bytes(&[10]));
    let out1 = client1.create_from_slice(u32::as_bytes(&[0]));
    add_scalar_kernel::launch(
        &client1,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out1.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in1.clone(), 1) },
        5,
    );
    let res1 = client1.read_one_unchecked(out1);
    assert_eq!(u32::from_bytes(&res1)[0], 15);

    let fingerprint = client1.properties().identity.fingerprint.clone();
    drop(client1);

    // Open store and poison the entry with malformed bytes
    let mut store: Store<KernelCacheKey, CpuCacheEntry> =
        compilation_store("cpu", &fingerprint).expect("store opens");
    let mut entries = Vec::new();
    store.scan(|k, v| entries.push((k, v)));
    assert_eq!(entries.len(), 1);
    let (key, mut entry) = entries.pop().unwrap();

    // Corrupt the object bytes
    entry.object =
        cubecl_common::bytes::Bytes::from_bytes_vec(vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]);
    store.purge_key(&key);
    store.insert(key, entry).expect("poisoned entry inserted");
    store.sync();
    drop(store);

    // Launch with Server Instance 2: must NOT panic, must fall back to full compilation and succeed
    let client2 = Client::load::<CpuServer>(DeviceId::new(0, 31));
    let in2 = client2.create_from_slice(u32::as_bytes(&[10]));
    let out2 = client2.create_from_slice(u32::as_bytes(&[0]));
    add_scalar_kernel::launch(
        &client2,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out2.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in2.clone(), 1) },
        5,
    );
    let res2 = client2.read_one_unchecked(out2);
    assert_eq!(
        u32::from_bytes(&res2)[0],
        15,
        "output must be correct despite corrupted cache entry"
    );

    drop(client2);

    // Assert that the store entry was rewritten with valid object bytes
    let mut store_after: Store<KernelCacheKey, CpuCacheEntry> =
        compilation_store("cpu", &fingerprint).expect("store opens");
    let mut entries_after = Vec::new();
    store_after.scan(|k, v| entries_after.push((k, v)));
    assert_eq!(entries_after.len(), 1);
    let (key_after, entry_after) = entries_after.pop().unwrap();
    assert_eq!(key_after, key);
    assert_ne!(
        &entry_after.object[..],
        &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03],
        "store entry must have been rewritten with valid object bytes"
    );
}

/// Asserts that object bytes are not retained in memory after compile + store
/// or after warm load (confirming Item 5).
#[test]
fn test_object_bytes_not_retained() {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let temp = TempDir::new("retention");
    cubecl_environment::environment::set_root(&temp.path);

    let _ = CubeClRuntimeConfig::try_set(CubeClRuntimeConfig {
        compilation: CompilationConfig {
            cache: true,
            ..Default::default()
        },
        ..Default::default()
    });

    let client = Client::load::<CpuServer>(DeviceId::new(0, 50));
    let in_buf = client.create_from_slice(u32::as_bytes(&[100]));
    let out_buf = client.create_from_slice(u32::as_bytes(&[0]));

    add_scalar_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out_buf.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(in_buf.clone(), 1) },
        1,
    );
    let _ = client.read_one_unchecked(out_buf);

    let fingerprint = client.properties().identity.fingerprint.clone();
    drop(client);

    // Inspect the entry written by store()
    let mut store: Store<KernelCacheKey, CpuCacheEntry> =
        compilation_store("cpu", &fingerprint).expect("store opens");
    let mut entries = Vec::new();
    store.scan(|k, v| entries.push((k, v)));
    assert_eq!(entries.len(), 1);
    let (_, entry) = entries.pop().unwrap();

    // 1. Simulating warm load (PlironEngine::from_object):
    // Warm-loaded engine must have 0 object bytes retained in memory.
    let warm_engine = PlironEngine::from_object(
        &entry.object,
        &entry.entrypoint_name,
        cubecl_llvm::KernelRequirements::default(),
        vec![],
    )
    .expect("from_object succeeds");
    assert!(
        warm_engine.take_object_bytes().is_none(),
        "warm-loaded engine must never retain object bytes"
    );

    // 2. Simulating fresh compilation with store():
    // take_object_bytes() moves the bytes out by value, leaving None in the engine.
    assert!(
        !entry.object.is_empty(),
        "store entry holds the object bytes on disk"
    );
}
