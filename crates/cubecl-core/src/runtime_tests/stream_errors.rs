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

use crate::{self as cubecl};
use alloc::string::{String, ToString};
use cubecl::prelude::*;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_common::bytes::Bytes;
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

/// A rejected launch belongs to the stream that made it, and to no neighbour
/// sharing its pooled stream.
pub fn test_a_rejected_launch_stays_on_its_own_stream<R: Runtime>(client: ComputeClient<R>) {
    let (launching, neighbour) = sharing_one_pooled_stream(1_000_001);

    let out = launching.executes(|| launch_rejected(&client, "attribution"));

    neighbour.executes(|| {
        client
            .flush()
            .expect("the neighbouring stream launched nothing, so it has nothing to report")
    });

    launching.executes(|| assert_rejected(&client, out, "attribution"));
}

/// A read is only as good as the work that wrote the buffer.
///
/// The rejection belongs to the stream that launched, so the reader's own flush
/// never sees it — and a read that does not consult the producer hands back the
/// buffer the failed launch never wrote.
pub fn test_a_read_surfaces_the_producers_rejection<R: Runtime>(client: ComputeClient<R>) {
    let (producer, reader) = sharing_one_pooled_stream(1_000_002);

    let out = producer.executes(|| launch_rejected(&client, "producer"));

    reader.executes(|| assert_rejected(&client, out, "producer"));
    // Reading the producer's error does not take it: the stream that made the
    // launch still reports it itself.
    producer.executes(|| {
        client
            .flush()
            .expect_err("the launching stream keeps its own rejection")
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

    let read = reader.executes(|| client.read_one(untouched).expect("nothing wrote to this one"));
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

/// A failed launch says nothing about the buffers it only reads.
///
/// A launch names its outputs when it fails, because those are the buffers
/// something was going to write and now nothing has. Its inputs it leaves
/// exactly as it found them — they hold whatever the stream that filled them
/// put there — so failing a read of one reports a kernel the reader never
/// launched, about memory that is perfectly good. `&mut` in the signature is
/// what separates the two.
pub fn test_a_failed_launch_leaves_the_buffers_it_only_reads_readable<R: Runtime>(
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
    // The input is untouched, and still says what its writer wrote.
    let read = reader.executes(|| {
        client
            .read_one(input)
            .expect("the failed launch never wrote this one, it only read it")
    });
    assert_eq!(u32::from_bytes(&read), &[7]);
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
            fn test_a_rejected_launch_stays_on_its_own_stream() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_rejected_launch_stays_on_its_own_stream::<
                    TestRuntime,
                >(client);
            }

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
            fn test_a_failed_launch_leaves_the_buffers_it_only_reads_readable() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_failed_launch_leaves_the_buffers_it_only_reads_readable::<
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
            fn test_a_buffer_the_failure_never_touched_reads_normally() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream_errors::test_a_buffer_the_failure_never_touched_reads_normally::<
                    TestRuntime,
                >(client);
            }
        }
    };
}
