mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, test_client};

use cubecl_runtime::server::CubeCount;
use cubecl_runtime::server::KernelArguments;
use cubecl_runtime::{local_tuner, tune::LocalTuner};
use dummy::*;

#[test_log::test]
fn created_resource_is_the_same_when_read() {
    let client = test_client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create_from_slice(&resource);

    let obtained_resource = client.read_one(resource_description).unwrap().to_vec();

    assert_eq!(resource, obtained_resource)
}

#[test_log::test]
fn empty_allocates_memory() {
    let client = test_client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read_one(resource_description).unwrap();

    assert_eq!(empty_resource.len(), 4);
}

// Dry runs are process-wide, so a test asserting that a launch really ran must
// not overlap one. `parallel` still runs alongside the other parallel tests; it
// only excludes the `serial` ones.
#[test_log::test]
#[serial_test::parallel]
fn execute_elementwise_addition() {
    let client = test_client(&DummyDevice);
    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);

    client.launch(
        Box::new(KernelTask::new(DummyElementwiseAddition)),
        CubeCount::Static(1, 1, 1),
        KernelArguments::new().with_buffers(vec![
            lhs.binding(),
            rhs.binding(),
            out.clone().binding(),
        ]),
    );

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
}

#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn autotune_basic_addition_execution() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_basic_addition_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([4, 5, 6]));
}

#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn autotune_basic_multiplication_execution() {
    static TUNER: LocalTuner<String, String> =
        local_tuner!("autotune_basic_multiplication_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::multiplication_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out).unwrap().to_vec();

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([0, 4, 8]));
}

/// A tuned pick belongs to the environment it was tuned under: switching
/// environments makes it unreachable, tuning again lands in the new one, and
/// switching back serves the persisted result through hydration rather than
/// re-tuning.
#[test_log::test]
#[cfg(all(feature = "std", autotune_persistence))]
#[serial_test::serial]
fn autotune_resets_when_the_environment_switches() {
    use cubecl_runtime::tune::{TuneCacheResult, Tuner};

    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(first.path());

    let client = test_client(&DummyDevice);
    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let set = dummy::addition_set(test_client(&DummyDevice), shapes);

    let handles = vec![
        client.create_from_slice(&[0, 1, 2]),
        client.create_from_slice(&[4, 4, 4]),
        client.empty(3),
    ];
    let key = set.generate_key(&handles);

    let tuner: Tuner<String> = Tuner::new("environment-switch", "device0");
    tuner.check_tune(
        &key,
        &handles,
        &set,
        || set.compute_checksum(),
        &client,
        None,
    );
    assert!(matches!(tuner.fastest(&key), TuneCacheResult::Hit { .. }));

    // The pick was tuned under `first`: after the switch it must not be
    // served, and tuning again fills `second`.
    cubecl_environment::environment::set_root(second.path());
    assert!(matches!(tuner.fastest(&key), TuneCacheResult::Miss));
    tuner.check_tune(
        &key,
        &handles,
        &set,
        || set.compute_checksum(),
        &client,
        None,
    );
    assert!(matches!(tuner.fastest(&key), TuneCacheResult::Hit { .. }));

    // Switching back serves `first`'s persisted result through hydration and
    // checksum validation, with no third tune.
    cubecl_environment::environment::set_root(first.path());
    assert!(matches!(tuner.fastest(&key), TuneCacheResult::Miss));
    let rehydrated = tuner.check_tune(
        &key,
        &handles,
        &set,
        || set.compute_checksum(),
        &client,
        None,
    );
    assert!(matches!(rehydrated, TuneCacheResult::Hit { .. }));
}

/// A throughput bound with a generous `time_limit` makes the tuner short-circuit: it
/// accepts the first candidate whose median is under the limit and never benchmarks the
/// rest. The set registers the slow+wrong kernel first, so a hit proves the faster `add`
/// was skipped rather than raced and lost.
#[test_log::test]
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[serial_test::serial]
fn autotune_bounds_short_circuit_accepts_first_within_limit() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_bounds_short_circuit");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        // time_limit = (1 / 1.0) / 1.0 = 1s, far above the ~few-ms slow kernel, so the
        // first candidate is already "close enough".
        dummy::bounded_addition_set_slow_first(client, shapes, 1.0, 1.0)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained = client.read_one(out).unwrap().to_vec();

    // The slow+wrong kernel copies lhs -> out. Getting it back means the tuner stopped
    // at the first candidate and never reached the faster, correct `add`.
    assert_eq!(obtained, vec![0, 1, 2]);
}

/// The mirror of the test above: an unreachable `time_limit` disqualifies every
/// candidate, so the tuner falls back to benchmarking the whole batch and the faster
/// `add` wins despite being registered second. This isolates the short-circuit as the
/// cause of the early exit, not the mere presence of a bound.
#[test_log::test]
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[serial_test::serial]
fn autotune_bounds_unreachable_limit_benchmarks_all() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_bounds_unreachable_limit");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        // time_limit = (1 / 1e12) / 1.0 ≈ 1ps, below any real median, so nothing qualifies.
        dummy::bounded_addition_set_slow_first(client, shapes, 1e12, 1.0)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained = client.read_one(out).unwrap().to_vec();

    assert_eq!(obtained, vec![4, 5, 6]);
}

/// `with_short_circuit(false)` disables early exit even when the bound is generous.
/// The slow+wrong kernel is first, but since short-circuit is off, the tuner benchmarks
/// all candidates and the faster correct `add` wins.
#[test_log::test]
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[serial_test::parallel]
fn autotune_short_circuit_disabled_benchmarks_all() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_short_circuit_disabled");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::bounded_addition_set_no_short_circuit(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained = client.read_one(out).unwrap().to_vec();

    // Short-circuit is disabled, so all candidates are benchmarked and the
    // faster correct `add` kernel wins despite the generous bound.
    assert_eq!(obtained, vec![4, 5, 6]);
}

/// 2-I1 — A panic inside a profiled closure surfaces at the `ComputeClient` caller as
/// the *original* panic (the issue's symptom), instead of an opaque `CallError`.
#[test_log::test]
#[cfg(feature = "std")]
fn profile_reraises_panic_from_profiled_closure() {
    let client = test_client(&DummyDevice);

    let reraised = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.profile(|| panic!("kernel boom"), "test")
    }));

    let payload = match reraised {
        Ok(_) => panic!("a panic in the profiled closure must surface at the caller"),
        Err(payload) => payload,
    };
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("kernel boom"),
        "the re-raised panic must carry the original message"
    );
}

/// 2-I2 — The success path through `profile` still returns `Ok` (guards against the
/// `unwrap_or_resume` swap turning a normal result into a panic).
#[test_log::test]
#[cfg(feature = "std")]
fn profile_returns_ok_on_success() {
    let client = test_client(&DummyDevice);

    let (value, _duration) = client
        .profile(|| 123u32, "ok")
        .expect("a successful profiled closure must return Ok");
    assert_eq!(value, 123);
}

/// 2-I3 — Design guard: the public `ComputeClient::exclusive` stays *recoverable* — a
/// task panic becomes `Err(ServerError::Generic)` (so autotune can skip a failing
/// candidate) rather than re-raising. The original message is still preserved in the
/// error string thanks to the `CallError` payload.
#[test_log::test]
#[cfg(feature = "std")]
fn exclusive_stays_recoverable_on_task_panic() {
    use cubecl_runtime::server::ServerError;

    let client = test_client(&DummyDevice);

    let result = client.exclusive(|| panic!("exclusive boom"));

    match result {
        Err(ServerError::Generic { reason, .. }) => assert!(
            reason.contains("exclusive boom"),
            "the recoverable error must carry the original message, got: {reason}"
        ),
        Err(other) => panic!("expected a recoverable ServerError::Generic, got: {other}"),
        Ok(()) => panic!("expected exclusive to return Err on a task panic, not Ok"),
    }
}

/// A tunable that rejects its own configuration fails identically on every call, so the
/// benchmark must stop at the first rejection rather than paying a profile round trip for
/// every warmup and sample before reporting it.
#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn autotune_stops_sampling_a_rejected_candidate() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_rejected_candidate");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_set = calls.clone();

    // The persistent cache outlives the process, so the key has to be new on every run for the
    // candidates to actually be benchmarked.
    let uid = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();

    let test_set = TUNER.init(move || {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set_with_rejected_candidate(client, shapes, uid.clone(), calls_set.clone())
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    // The rejected candidate is dropped after its first failure, and the surviving `add`
    // kernel still wins the tuning.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(client.read_one(out).unwrap().to_vec(), vec![4, 5, 6]);
}

/// A candidate whose kernel fails to compile is handled lazily, with no unwinding
/// anywhere: the server records the launch failure and returns it at `end_profile`, the
/// tuner drops the candidate on that error, the surviving kernel wins, and the device
/// keeps serving afterwards.
#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn autotune_skips_a_candidate_that_fails_compilation() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_failing_compilation");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let uid = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();

    let test_set = TUNER.init(move || {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set_with_failing_compilation(client, shapes, uid.clone())
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    // The failing candidate was skipped on the lazily returned error and `add` won.
    assert_eq!(client.read_one(out).unwrap().to_vec(), vec![4, 5, 6]);

    // The failure never became a panic: the device keeps serving.
    let after = client
        .exclusive(|| 42)
        .expect("the device must keep serving after a candidate failed to compile");
    assert_eq!(after, 42);
}

/// The round robin end to end, which the unit tests around it cannot reach: a candidate far
/// enough behind has to stop being sampled partway through, while the ones still in contention
/// keep going and the fastest of them wins.
///
/// Skipped unless the adaptive scheduler is the strategy in force, since a fixed-count pass
/// samples every candidate the same number of times by design.
#[test_log::test]
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[serial_test::serial]
fn autotune_stops_sampling_an_eliminated_candidate() {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let bench = CubeClRuntimeConfig::get().autotune.bench.clone();
    if !bench.adaptive {
        return;
    }
    let (min_samples, max_samples) = bench.samples();

    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_eliminated_candidate");

    let client = test_client(&DummyDevice);

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs, rhs, out.clone()];

    let fast_calls = Arc::new(AtomicUsize::new(0));
    let slow_calls = Arc::new(AtomicUsize::new(0));
    let fast_set = fast_calls.clone();
    let slow_set = slow_calls.clone();

    let uid = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();

    let test_set = TUNER.init(move || {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set_with_slow_candidate(
            client,
            shapes,
            uid.clone(),
            fast_set.clone(),
            slow_set.clone(),
        )
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let fast = fast_calls.load(Ordering::Relaxed);
    let slow = slow_calls.load(Ordering::Relaxed);

    // Every candidate is warmed up once and sampled at least to the elimination floor, so the
    // slow one cannot have been dropped before it had the evidence against it.
    assert!(
        slow > min_samples,
        "the slow candidate was dropped before it earned it: {slow} calls"
    );
    // A full pass is one warmup plus the ceiling. The slow candidate must fall short of that,
    // and short of what the survivors spent, or nothing was eliminated at all.
    assert!(
        slow < max_samples + 1,
        "the slow candidate was sampled to the ceiling: {slow} calls"
    );
    assert!(
        slow < fast,
        "the slow candidate kept pace with the survivors: {slow} vs {fast} calls"
    );

    // The fast kernel wins, so the output is a real addition rather than the slow kernel's copy.
    assert_eq!(client.read_one(out).unwrap().to_vec(), vec![4, 5, 6]);
}

/// A dry run drops an ordinary launch: the server still compiles the kernel,
/// exactly as it would otherwise, and then never runs it.
///
/// This is the mode's whole promise and its whole hazard in one assertion — a
/// pass under it runs for the shapes it provokes, and anything it reads back
/// is meaningless.
#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn a_dry_run_drops_an_ordinary_launch() {
    use cubecl_runtime::dry_run::DryRun;

    let client = test_client(&DummyDevice);
    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.create_from_slice(&[9, 9, 9]);

    let add = |out: &cubecl_runtime::server::Handle| {
        client.launch(
            Box::new(KernelTask::new(DummyElementwiseAddition)),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new().with_buffers(vec![
                lhs.clone().binding(),
                rhs.clone().binding(),
                out.clone().binding(),
            ]),
        );
    };

    {
        let _dry_run = DryRun::new();
        add(&out);

        assert_eq!(
            client.read_one(out.clone()).unwrap().to_vec(),
            Vec::from([9, 9, 9]),
            "the launch was compiled and then dropped, so the output is untouched"
        );
    }

    // The very same launch runs once the mode is off: nothing was poisoned by
    // having been skipped, and the compiled artifact is reused.
    add(&out);

    assert_eq!(client.read_one(out).unwrap().to_vec(), Vec::from([4, 5, 6]));
}

/// The exception that makes the mode worth having: autotune still executes,
/// because its launches *are* the measurement.
///
/// Tuning happens inside the dry run; the winner is then executed outside it. A
/// tuner whose candidates had all been skipped would have nothing to tell them
/// apart, and the slow kernel — which writes `[0, 1, 2]` — would win as often
/// as not.
#[test_log::test]
#[cfg(feature = "std")]
#[serial_test::serial]
fn a_dry_run_still_autotunes() {
    use cubecl_runtime::dry_run::DryRun;

    static TUNER: LocalTuner<String, String> = local_tuner!("a_dry_run_still_autotunes");

    let client = test_client(&DummyDevice);
    let test_set = TUNER.init(|| {
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set(test_client(&DummyDevice), shapes)
    });

    let lhs = client.create_from_slice(&[0, 1, 2]);
    let rhs = client.create_from_slice(&[4, 4, 4]);
    let out = client.empty(3);

    {
        let _dry_run = DryRun::new();
        TUNER.execute(
            &"test".to_string(),
            &client,
            test_set.clone(),
            vec![lhs.clone(), rhs.clone(), out.clone()],
        );
    }

    // Cached now, so this is the fast path: it executes the winner and nothing
    // else.
    TUNER.execute(
        &"test".to_string(),
        &client,
        test_set,
        vec![lhs, rhs, out.clone()],
    );

    assert_eq!(
        client.read_one(out).unwrap().to_vec(),
        Vec::from([4, 5, 6]),
        "the candidates were measured inside the dry run, so the fast one won"
    );
}
