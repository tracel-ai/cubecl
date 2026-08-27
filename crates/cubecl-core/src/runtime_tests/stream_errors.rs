//! What a read is owed when the work that was supposed to write a buffer never
//! ran.
//!
//! Errors are lazy: a failed launch is queued and surfaces at the next flush,
//! sync or profile end. That leaves two questions every backend has to answer
//! the same way, and these tests are where the answers are pinned down.
//!
//! *Who reports it* — logical streams are folded onto the pooled ones with
//! `id % max_streams`, so two of them share a backend stream. A neighbour that
//! drained somebody else's rejection would fail on a kernel it never launched,
//! while the stream that did launch it read back an untouched buffer as if all
//! was well.
//!
//! *Were these bytes ever written* — which no stream id can answer, because a
//! handle names where a buffer was **created** and nothing re-tags it.
//!
//! And *for how long* — the answers end at different times. The report is owed
//! once and the flush that makes it is usually the launching stream's own,
//! which comes long before anyone reads; the buffer stays stale until something
//! fills it, from the host or from a launch that runs.

use crate::{self as cubecl};
use alloc::string::{String, ToString};
use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_runtime::server::Handle;

/// Fills every element with `value`, so a buffer says which write reached it
/// last.
#[cube(launch)]
pub fn fill(out: &mut [u32], #[comptime] value: u32) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = value;
    }
}

/// A launch the compiler is guaranteed to refuse, whatever the target, and
/// before anything touches a buffer.
#[cube(launch_unchecked)]
pub fn rejected(out: &mut [u32], #[comptime] reason: String) {
    push_validation_error(reason);
    out[0] = 1u32;
}

/// Copies input to output, so a failure can travel one hop downstream.
#[cube(launch)]
pub fn copy(input: &[u32], out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// The same refusal, with an input the kernel only reads alongside the output
/// it writes.
#[cube(launch_unchecked)]
pub fn rejected_with_input(input: &[u32], out: &mut [u32], #[comptime] reason: String) {
    push_validation_error(reason);
    out[0] = input[0];
}

/// Two logical streams landing on one pooled stream.
///
/// `seed` is far above the ids [`StreamId::current`] hands out per thread,
/// which are small and sequential: a test that pins a low id shares it outright
/// with whichever sibling test's thread was assigned the same number, and then
/// legitimately drains that sibling's errors.
fn sharing_one_pooled_stream(seed: u64) -> (StreamId, StreamId) {
    let max_streams = CubeClRuntimeConfig::get().streaming.max_streams as u64;
    (
        StreamId { value: seed },
        StreamId {
            value: seed + max_streams,
        },
    )
}

fn launch_rejected_into<R: Runtime>(client: &ComputeClient<R>, out: Handle, reason: &str) {
    unsafe {
        rejected::launch_unchecked::<R>(
            &client.clone(),
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(out, 1),
            reason.to_string(),
        )
    };
}

fn launch_rejected<R: Runtime>(client: &ComputeClient<R>, reason: &str) -> Handle {
    let out = client.empty(core::mem::size_of::<u32>());
    launch_rejected_into(client, out.clone(), reason);
    out
}

fn assert_rejected<R: Runtime>(client: &ComputeClient<R>, out: Handle, reason: &str) {
    let err = client
        .read_one(out)
        .expect_err("the kernel pushed a validation error, the launch must fail")
        .to_string();
    assert!(
        err.contains(reason),
        "the read must report the launch that never wrote the buffer, got: {err}"
    );
}

/// A read is only as good as the work that wrote the buffer.
///
/// The rejection belongs to the stream that launched, so the reader's own flush
/// never sees it — and a read that does not consult the producer hands back the
/// buffer the failed launch never wrote.
pub fn test_a_read_surfaces_the_producers_rejection<R: Runtime>(client: ComputeClient<R>) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_002);

    let out = producer.executes(|| launch_rejected(&client, "producer"));

    reader.executes(|| assert_rejected(&client, out.clone(), "producer"));
    // Reading the failure does not clear it: the claim lives on the buffer
    // until something writes it. And it is nobody's to *report* — a launch
    // failure is not the flush's business any more.
    producer.executes(|| {
        client
            .flush()
            .expect("a launch failure is not the flush's to report");
        assert_rejected(&client, out, "producer");
    });
}

/// The case no stream id can answer: the buffer is allocated on the stream that
/// reads it back, and written by a launch that failed on another.
///
/// `Handle::stream` names where a buffer was created and nothing re-tags it, so
/// the allocation and the read both name the reader — which launched nothing
/// and has nothing queued. Only the buffer itself connects the read to the
/// launch that never ran.
pub fn test_a_read_surfaces_a_rejection_on_a_buffer_it_allocated_itself<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_003);

    let out = reader.executes(|| client.empty(core::mem::size_of::<u32>()));
    producer.executes(|| launch_rejected_into(&client, out.clone(), "foreign-write"));

    reader.executes(|| assert_rejected(&client, out, "foreign-write"));
}

/// A failure on one buffer is not a reason to refuse a read of another.
///
/// The queued error stays until its own stream flushes — under a per-task
/// stream policy, possibly never. A read that failed on the mere presence of an
/// error somewhere would refuse every unrelated buffer for as long as that
/// entry sat there.
pub fn test_a_buffer_the_failure_never_touched_reads_normally<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_004);

    let untouched = reader.executes(|| client.create_from_slice(u32::as_bytes(&[7u32])));
    producer.executes(|| launch_rejected(&client, "unrelated"));

    let read = reader.executes(|| {
        client
            .read_one(untouched)
            .expect("nothing wrote to this one")
    });
    assert_eq!(u32::from_bytes(&read), &[7]);
}

/// A host write lands after the work already queued on the stream that owns the
/// buffer, not whenever that stream happens to be flushed.
///
/// The two streams queue independently, so a write registered with no dependency
/// on the handle's stream is ordered only by whichever queue the next flush
/// drains first — and a launch that was enqueued *before* the write can land
/// *after* it, overwriting bytes the caller had every reason to believe were the
/// last word.
pub fn test_a_write_lands_after_the_work_queued_on_its_buffers_stream<R: Runtime>(
    client: ComputeClient<R>,
) {
    // Two ids far enough apart to land on different pooled streams for any
    // `max_streams`: one queue has to be aligned against the other for the
    // ordering to mean anything.
    let owner = StreamId { value: 2_000_001 };
    let writer = StreamId { value: 2_000_002 };
    let len = 4;

    let buf = owner.executes(|| {
        let buf = client.empty(len * core::mem::size_of::<u32>());
        fill::launch::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(len as u32),
            unsafe { BufferArg::from_raw_parts(buf.clone(), len) },
            1,
        );
        buf
    });

    writer.executes(|| {
        client.write(
            &buf,
            Bytes::from_bytes_vec(u32::as_bytes(&[9u32; 4]).to_vec()),
        )
    });

    let read = writer.executes(|| client.read_one(buf).expect("both writes succeeded"));
    assert_eq!(
        u32::from_bytes(&read),
        &[9, 9, 9, 9],
        "the host write was issued last, so it has to land last"
    );
}

/// A launch that never compiled taints everything it was given, the buffers
/// it would only have read included.
///
/// The read set and the write set are the compiled kernel's own answer, and a
/// kernel that failed to compile never gave one — there is no IR to ask. So
/// every buffer the launch was handed is claimed by the failure: naming the
/// input fails a read that would have been fine, loudly, where guessing from
/// the signature would be the write mask this design deleted. A launch that
/// fails *after* compilation stages only what the kernel writes, so its
/// read-only inputs stay readable — the property the write scope's own tests
/// pin, since no portable kernel fails between compilation and enqueue on
/// every backend.
pub fn test_a_launch_that_never_compiled_taints_everything_it_was_given<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_005);

    let (input, out) = producer.executes(|| {
        let input = client.create_from_slice(u32::as_bytes(&[7u32]));
        let out = client.empty(core::mem::size_of::<u32>());
        unsafe {
            rejected_with_input::launch_unchecked::<R>(
                &client.clone(),
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(input.clone(), 1),
                BufferArg::from_raw_parts(out.clone(), 1),
                "read-only-input".to_string(),
            )
        };
        (input, out)
    });

    // The output is what the launch was going to write, so it is unreadable.
    reader.executes(|| assert_rejected(&client, out, "read-only-input"));
    // With no compiled kernel to say otherwise, the input is claimed too.
    reader.executes(|| assert_rejected(&client, input.clone(), "read-only-input"));

    // And writing it again is the recovery, exactly as for an output.
    producer.executes(|| {
        client.write(
            &input,
            Bytes::from_bytes_vec(u32::as_bytes(&[7u32]).to_vec()),
        )
    });
    let read = reader.executes(|| {
        client
            .read_one(input)
            .expect("a rewritten buffer reads again")
    });
    assert_eq!(u32::from_bytes(&read), &[7]);
}

/// A report is owed once; the bytes stay stale until something writes them.
///
/// The flush that reports a launch failure is usually the launching stream's
/// own, and flushing has nothing to do with the claim: the bytes stay stale
/// until something writes them, however many flushes and reads come between.
pub fn test_a_buffer_stays_unreadable_after_the_failure_was_reported<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_007);

    let out = producer.executes(|| launch_rejected(&client, "reported-once"));

    producer.executes(|| {
        client
            .flush()
            .expect("a launch failure is not the flush's to report");
        client.flush().expect("on any flush, however many");
    });

    // Reported is not written.
    reader.executes(|| assert_rejected(&client, out.clone(), "reported-once"));
    producer.executes(|| assert_rejected(&client, out, "reported-once"));
}

/// A buffer a failed launch left stale is sound again once something fills it.
///
/// Filling it from the host is how a caller recovers when it has the bytes
/// already, and it happens before the flush that reports the failure as often
/// as after — so a claim that only listened for writes from its report onward
/// would refuse the buffer for good.
pub fn test_a_host_write_makes_a_stale_buffer_readable_again<R: Runtime>(client: ComputeClient<R>) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_008);

    let out = producer.executes(|| {
        let out = client.empty(core::mem::size_of::<u32>());
        launch_rejected_into(&client, out.clone(), "recovered");
        out
    });

    // Nothing has written it yet.
    reader.executes(|| assert_rejected(&client, out.clone(), "recovered"));

    producer.executes(|| {
        client.write(
            &out,
            Bytes::from_bytes_vec(u32::as_bytes(&[42u32]).to_vec()),
        )
    });

    let read = reader.executes(|| {
        client
            .read_one(out)
            .expect("the host write filled what the launch never did")
    });
    assert_eq!(u32::from_bytes(&read), &[42]);

    // And nothing was left owing a report anywhere.
    producer.executes(|| {
        client
            .flush()
            .expect("a launch failure is not the flush's to report")
    });
}

/// A relaunch into the same buffer is the other way out, and it must not be
/// refused on its way in: a pure output is not read, so the launch that would
/// repair a buffer is never the one skipped for its being broken.
pub fn test_a_relaunch_makes_a_stale_buffer_readable_again<R: Runtime>(client: ComputeClient<R>) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_009);
    let len = 4;

    let out = producer.executes(|| {
        let out = client.empty(len * core::mem::size_of::<u32>());
        launch_rejected_into(&client, out.clone(), "relaunched");
        fill::launch::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(len as u32),
            unsafe { BufferArg::from_raw_parts(out.clone(), len) },
            3,
        );
        out
    });

    let read = reader.executes(|| {
        client
            .read_one(out)
            .expect("the second launch wrote what the first never did")
    });
    assert_eq!(u32::from_bytes(&read), &[3; 4]);
}

/// A launch whose input carries a failure is skipped, and its outputs take
/// the failure that stopped it — so a read anywhere downstream fails on the
/// root cause instead of copying out bytes nothing computed.
///
/// This is the case the whole model exists for. Every backend used to clear
/// its outputs on a successful launch without looking at its inputs, so the
/// direct read of a buffer nothing wrote was caught and everything computed
/// *from* one was missed — in a fused stack, nearly everything that matters.
pub fn test_a_read_downstream_of_a_failure_fails_on_the_root_cause<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_011);

    let launch_copy = |input: &Handle, out: &Handle| {
        copy::launch::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(input.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
        );
    };

    let (out1, out2, out3) = producer.executes(|| {
        let out1 = launch_rejected(&client, "root-cause");
        let out2 = client.empty(core::mem::size_of::<u32>());
        let out3 = client.empty(core::mem::size_of::<u32>());
        // Two hops of valid kernels reading what the rejection left stale:
        // neither runs, and everything downstream carries the rejection.
        launch_copy(&out1, &out2);
        launch_copy(&out2, &out3);
        (out1, out2, out3)
    });

    // Two hops down, the read reports the root cause — and the path to it:
    // each hop the launch that was skipped and the buffer that stopped it.
    let err = reader.executes(|| {
        client
            .read_one(out3)
            .expect_err("the copies never ran, so nothing wrote this buffer")
            .to_string()
    });
    assert!(
        err.contains("root-cause"),
        "the root failure travels the whole chain: {err}"
    );
    assert!(
        err.matches("skipped `").count() == 2,
        "both skipped hops are named on the way to the root: {err}"
    );
    assert!(
        err.contains("failure #"),
        "the failure id ties every read of this failure together: {err}"
    );

    // The middle and root buffers report the same failure.
    reader.executes(|| assert_rejected(&client, out2, "root-cause"));
    reader.executes(|| assert_rejected(&client, out1, "root-cause"));
}

/// `check` answers whether the bytes can be trusted without reading them and
/// without a barrier — one lookup, so a caller can recover per tensor instead
/// of tearing down a device.
pub fn test_check_answers_without_a_read<R: Runtime>(client: ComputeClient<R>) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_013);

    let (stale, clean) = producer.executes(|| {
        let stale = launch_rejected(&client, "checked-not-read");
        let clean = client.create_from_slice(u32::as_bytes(&[5u32]));
        (stale, clean)
    });

    reader.executes(|| {
        client
            .check(&clean)
            .expect("a buffer whose writer succeeded checks clean");
        let err = client
            .check(&stale)
            .expect_err("the rejected launch never wrote this one")
            .to_string();
        assert!(
            err.contains("checked-not-read"),
            "the check names the failure a read would have named: {err}"
        );
    });

    // And the check took nothing: a read still reports the same failure.
    reader.executes(|| assert_rejected(&client, stale, "checked-not-read"));
}

/// An output that aliases an input writes that input in place, so a failure
/// leaves the aliased buffer unwritten however its own argument was declared.
///
/// This is the case that makes narrowing dangerous rather than merely wrong:
/// the aliased buffer arrives through a `&[T]` parameter, so a mask built from
/// the signature alone calls it read-only and leaves it unnamed — and a read of
/// the one buffer an in-place kernel exists to produce hands back the bytes
/// that were there before, with no error anywhere.
pub fn test_a_failed_in_place_launch_names_the_buffer_it_aliased<R: Runtime>(
    client: ComputeClient<R>,
) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_006);

    let inout = producer.executes(|| {
        let inout = client.create_from_slice(u32::as_bytes(&[7u32]));
        unsafe {
            rejected_with_input::launch_unchecked::<R>(
                &client.clone(),
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(inout.clone(), 1),
                // The output is the input: nothing new is registered, and the
                // kernel writes the buffer that came in through `&[u32]`.
                BufferArg::alias(0, 1),
                "aliased-output".to_string(),
            )
        };
        inout
    });

    reader.executes(|| assert_rejected(&client, inout, "aliased-output"));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_stream_errors {
    () => {
        mod stream_errors {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_a_read_surfaces_the_producers_rejection() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_read_surfaces_the_producers_rejection::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_read_surfaces_a_rejection_on_a_buffer_it_allocated_itself() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_read_surfaces_a_rejection_on_a_buffer_it_allocated_itself::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_launch_that_never_compiled_taints_everything_it_was_given() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_launch_that_never_compiled_taints_everything_it_was_given::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_read_downstream_of_a_failure_fails_on_the_root_cause() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_read_downstream_of_a_failure_fails_on_the_root_cause::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_check_answers_without_a_read() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_check_answers_without_a_read::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_failed_in_place_launch_names_the_buffer_it_aliased() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_failed_in_place_launch_names_the_buffer_it_aliased::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_write_lands_after_the_work_queued_on_its_buffers_stream() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_write_lands_after_the_work_queued_on_its_buffers_stream::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_buffer_stays_unreadable_after_the_failure_was_reported() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_buffer_stays_unreadable_after_the_failure_was_reported::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_host_write_makes_a_stale_buffer_readable_again() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_host_write_makes_a_stale_buffer_readable_again::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_relaunch_makes_a_stale_buffer_readable_again() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_relaunch_makes_a_stale_buffer_readable_again::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_buffer_the_failure_never_touched_reads_normally() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_buffer_the_failure_never_touched_reads_normally::<
                    TestRuntime,
                >(client);
            }
        }
    };
}
