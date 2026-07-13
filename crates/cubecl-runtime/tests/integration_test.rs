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

#[test_log::test]
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

/// A throughput bound with a generous `time_limit` makes the tuner short-circuit: it
/// accepts the first candidate whose median is under the limit and never benchmarks the
/// rest. The set registers the slow+wrong kernel first, so a hit proves the faster `add`
/// was skipped rather than raced and lost.
#[test_log::test]
#[cfg(all(feature = "std", not(target_family = "wasm")))]
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
