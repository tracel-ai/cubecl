//! What the caches actually cost on the paths they sit on.
//!
//! Reads are meant to be a plain hash lookup, so they are measured against a
//! bare `HashMap` to expose any overhead the store adds. Opening and inserting
//! do touch the database, and are measured separately so the two are never
//! confused.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::{KvStore, KvStoreOptions};

/// Realistic for an autotune namespace: a few hundred tuned shapes.
const ENTRIES: usize = 512;

/// An autotune-shaped key, long enough that hashing is not free.
fn key(index: usize) -> String {
    format!("matmul-lhs=f16-rhs=f16-out=f32-shape={index}x{index}x{index}")
}

fn warm(root: &std::path::Path) -> KvStore<String, u32> {
    cubecl_environment::environment::set_root(root);
    let mut store = KvStore::<String, u32>::open("device0/matmul", KvStoreOptions::default());

    for index in 0..ENTRIES {
        store.insert(key(index), index as u32).ok();
    }

    store
}

fn read_path(criterion: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let store = warm(dir.path());

    let mut baseline: HashMap<String, u32> = HashMap::default();
    for index in 0..ENTRIES {
        baseline.insert(key(index), index as u32);
    }

    let keys: Vec<String> = (0..ENTRIES).map(key).collect();

    let mut group = criterion.benchmark_group("read");
    group.bench_function("kv_store_hit", |bencher| {
        let mut index = 0;
        bencher.iter(|| {
            index = (index + 1) % ENTRIES;
            black_box(store.get(black_box(&keys[index])))
        })
    });
    group.bench_function("hashmap_hit", |bencher| {
        let mut index = 0;
        bencher.iter(|| {
            index = (index + 1) % ENTRIES;
            black_box(baseline.get(black_box(&keys[index])))
        })
    });

    let missing = key(ENTRIES + 1);
    group.bench_function("kv_store_miss", |bencher| {
        bencher.iter(|| black_box(store.get(black_box(&missing))))
    });
    group.finish();
}

fn write_path(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("write");

    // A repeat insert of a value the store already holds: the common autotune
    // case, and the one that must not reach the database. The key is built
    // outside the loop, so this measures the store rather than `format!`.
    let dir = tempfile::tempdir().unwrap();
    let mut store = warm(dir.path());
    let known = key(0);
    group.bench_function("kv_store_insert_known", |bencher| {
        bencher.iter(|| store.insert(black_box(known.clone()), black_box(0)))
    });

    // The same, minus the unavoidable key clone `insert` needs to take
    // ownership, so the two together show what the store itself costs.
    group.bench_function("key_clone_only", |bencher| {
        bencher.iter(|| black_box(known.clone()))
    });

    // A genuinely new entry, which commits a transaction to the database.
    // Batched so the key is built outside the timed section, and so every
    // iteration really does insert a key the store has never seen: reusing
    // keys would silently measure the in-memory path instead.
    let dir = tempfile::tempdir().unwrap();
    let mut store = warm(dir.path());
    let mut index = ENTRIES;
    group.bench_function("kv_store_insert_new", |bencher| {
        bencher.iter_batched(
            || {
                index += 1;
                key(index)
            },
            |fresh| store.insert(black_box(fresh), 0),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// The kernel-launch path: a memoized blob lookup, which goes through a
/// `RefCell` rather than a plain map.
fn blob_path(criterion: &mut Criterion) {
    use cubecl_environment::bytes::Bytes;
    use cubecl_environment::persistence::blob::BlobStore;

    let dir = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(dir.path());
    let mut store = BlobStore::<String, Bytes>::new("spirv_device0", KvStoreOptions::default());

    let keys: Vec<String> = (0..ENTRIES).map(key).collect();
    for key in &keys {
        store
            .insert(key.clone(), Bytes::from_bytes_vec(std::vec![7u8; 4096]))
            .ok();
    }

    let mut group = criterion.benchmark_group("blob");
    group.bench_function("blob_store_memoized_hit", |bencher| {
        let mut index = 0;
        bencher.iter(|| {
            index = (index + 1) % ENTRIES;
            black_box(store.get(black_box(&keys[index])))
        })
    });
    group.finish();
}

fn open_path(criterion: &mut Criterion) {
    // Startup cost: one scan of the namespace, decoded into memory. This is
    // the disk path, paid once per store, so it is measured at two sizes to
    // show how it scales with a warm cache.
    let mut group = criterion.benchmark_group("open");

    for entries in [512usize, 4096] {
        let dir = tempfile::tempdir().unwrap();
        {
            cubecl_environment::environment::set_root(dir.path());
            let mut store =
                KvStore::<String, u32>::open("device0/matmul", KvStoreOptions::default());
            for index in 0..entries {
                store.insert(key(index), index as u32).ok();
            }
        }

        group.bench_function(format!("kv_store_open_{entries}"), |bencher| {
            bencher.iter(|| {
                cubecl_environment::environment::set_root(dir.path());
                let store =
                    KvStore::<String, u32>::open("device0/matmul", KvStoreOptions::default());
                black_box(store.len())
            })
        });
    }

    group.finish();
}

/// What a byte-level cache would cost: a decoder run on every read, rather
/// than once per key ever.
fn decode_cost(criterion: &mut Criterion) {
    use cubecl_environment::bytes::Bytes;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone, Debug)]
    struct TuneValue {
        fastest_index: usize,
        timings: Vec<u64>,
    }

    let small = TuneValue {
        fastest_index: 3,
        timings: (0..8).collect(),
    };
    let small_bytes = {
        let mut buffer = Vec::new();
        ciborium::ser::into_writer(&small, &mut buffer).unwrap();
        buffer
    };

    let kernel = Bytes::from_bytes_vec(std::vec![7u8; 4096]);
    let kernel_bytes = {
        let mut buffer = Vec::new();
        ciborium::ser::into_writer(&kernel, &mut buffer).unwrap();
        buffer
    };

    let mut group = criterion.benchmark_group("decode");
    group.bench_function("autotune_value", |bencher| {
        bencher.iter(|| {
            let value: TuneValue =
                ciborium::de::from_reader(black_box(small_bytes.as_slice())).unwrap();
            black_box(value)
        })
    });
    group.bench_function("kernel_4kib", |bencher| {
        bencher.iter(|| {
            let value: Bytes =
                ciborium::de::from_reader(black_box(kernel_bytes.as_slice())).unwrap();
            black_box(value)
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    read_path,
    write_path,
    blob_path,
    open_path,
    decode_cost
);
criterion_main!(benches);
