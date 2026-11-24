//! A pass that discards its launches still hears that a kernel the work
//! reached can never compile, through the collector of the override it runs
//! under; an autotune candidate that fails to compile stays the race's own.
//!
//! Every test here opens a [`ProcessModeOverride`], which applies to the
//! whole process: they run from a test binary of their own, one at a time
//! ([`testgen_compile_only_failures`](crate::testgen_compile_only_failures)),
//! never beside the other runtime tests, whose launches it would discard.

use crate::{self as cubecl};
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use cubecl::prelude::*;
use cubecl_runtime::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;
use cubecl_runtime::tune::{LocalTuner, Tunable, TunableSet};

use super::launch::kernel_panicking_in_expansion;

/// What the panic of [`kernel_panicking_in_expansion`] says.
const REFUSAL: &str = "this kernel refuses to expand";

fn launch_expanding(client: &Client, output: &Handle, refuse: bool) {
    kernel_panicking_in_expansion::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
        refuse,
    );
}

/// A kernel the work launches under either process mode that discards
/// launches, and that panics while it expands, is listed among the
/// collector's work failures with the panic's message: queued and compiled
/// with its batch under `CompileOnly`, or compiled at its launch under
/// `CompileAndAutotune`. Nothing ran, so the output reads as it was.
pub fn test_compile_only_failure_is_reported<R: Runtime>(client: Client) {
    for mode in [ProcessMode::CompileOnly, ProcessMode::CompileAndAutotune] {
        let output = client.create_from_slice(u32::as_bytes(&[7]));
        let collector = StatisticsCollector::new();
        let discarding = ProcessModeOverride::new(mode, &collector);
        launch_expanding(&client, &output, true);
        client.compile_queued();
        // The server settles the launch on its own thread: a read of a buffer
        // the launch never touched waits for it without depending on it.
        let _ = client.read_one(client.create_from_slice(u32::as_bytes(&[0])));
        core::mem::drop(discarding);

        let failures = collector.work_failures();
        assert_eq!(failures.len(), 1, "{mode:?}: {failures:?}");
        let report = format!("{}", failures[0].error);
        assert!(
            report.contains(REFUSAL),
            "{mode:?}: the failure keeps the expansion's panic, got: {report}"
        );
        let actual = client
            .read_one(output)
            .expect("a discarded launch fails no read");
        assert_eq!(u32::from_bytes(&actual), &[7], "{mode:?}: nothing ran");
    }
}

type Inputs = (Client, Handle);

/// The autotune of a key, unique to this run so no persistent cache answers
/// it, between a candidate whose kernel panics while it expands and one that
/// writes 1.
fn tune_set(key: String) -> TunableSet<String, Inputs, ()> {
    let candidate = |refuse: bool| {
        move |(client, output): Inputs| -> Result<(), String> {
            launch_expanding(&client, &output, refuse);
            Ok(())
        }
    };
    TunableSet::new_cloning_inputs(move |_: &Inputs| key.clone())
        .with(Tunable::new("refuses_to_expand", candidate(true)))
        .with(Tunable::new("writes_one", candidate(false)))
}

/// A build as an environment build runs it — a `CompileOnly` pass, then a
/// `CompileAndAutotune` pass, one collector — over an autotune whose first
/// candidate panics while it expands: the race drops that candidate and
/// counts it failed, the work lists nothing, and the pick, run for real
/// afterwards, is the candidate that works.
pub fn test_autotune_failing_candidate_reports_nothing<R: Runtime>(client: Client) {
    static TUNER: LocalTuner<String, String> = LocalTuner::new("compile_only_failures");
    let key = format!(
        "failing-candidate-{:?}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
    );
    let set = Arc::new(tune_set(key));
    let id = String::from("device");
    let output = client.create_from_slice(u32::as_bytes(&[7]));
    let inputs: Inputs = (client.clone(), output.clone());

    let collector = StatisticsCollector::new();
    for mode in [ProcessMode::CompileOnly, ProcessMode::CompileAndAutotune] {
        let pass = ProcessModeOverride::new(mode, &collector);
        TUNER.execute(&id, &client, set.clone(), inputs.clone());
        client.compile_queued();
        let _ = client.read_one(client.create_from_slice(u32::as_bytes(&[0])));
        core::mem::drop(pass);
    }

    let failures = collector.work_failures();
    assert!(
        failures.is_empty(),
        "a candidate the race dropped is no failure of the work: {failures:?}"
    );
    let compilation = collector.statistics().compilation;
    assert!(
        compilation.failed_count >= 1,
        "the failing candidate was compiled and failed: {compilation:?}"
    );

    TUNER.execute(&id, &client, set, inputs);
    let actual = client.read_one(output).expect("the pick runs");
    assert_eq!(
        u32::from_bytes(&actual),
        &[1],
        "the pick is the candidate that works"
    );
}

/// The tests of a pass that discards its launches, for a test binary of
/// their own: each opens a process-wide override, so they take a lock, and
/// nothing else may run in the binary beside them.
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_compile_only_failures {
    () => {
        static COMPILE_ONLY_FAILURES: std::sync::Mutex<()> = std::sync::Mutex::new(());

        #[test]
        fn test_compile_only_failure_is_reported() {
            let lock = COMPILE_ONLY_FAILURES
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::compile_only_failures::test_compile_only_failure_is_reported::<
                TestRuntime,
            >(client);
            core::mem::drop(lock);
        }

        #[test]
        fn test_autotune_failing_candidate_reports_nothing() {
            let lock = COMPILE_ONLY_FAILURES
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::compile_only_failures::test_autotune_failing_candidate_reports_nothing::<
                TestRuntime,
            >(client);
            core::mem::drop(lock);
        }
    };
}
