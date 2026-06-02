use std::sync::Arc;

use crate::compute::threadpool::global_buffer::GlobalBuffer;
use crate::compute::threadpool::thread_buffer::ThreadBuffer;

use super::shared::Task;
use super::shared::task;

fn make_thread_buffers<const NB_STREAM: usize, const CAPACITY: usize>(
    nb_threads: usize,
) -> Arc<[spin::Mutex<ThreadBuffer<Task, NB_STREAM, CAPACITY>>]> {
    let global = Arc::new(spin::Mutex::new(GlobalBuffer::new()));
    let buffers = (0..nb_threads)
        .map(|index| spin::Mutex::new(ThreadBuffer::new(index, global.clone())))
        .collect::<Vec<_>>();
    let buffers: Arc<[spin::Mutex<ThreadBuffer<Task, NB_STREAM, CAPACITY>>]> = Arc::from(buffers);

    for thread_buffer in buffers.iter() {
        thread_buffer.lock().set_global_queue(buffers.clone());
    }

    buffers
}

#[test]
fn pops_local_before_global() {
    let global = Arc::new(spin::Mutex::new(GlobalBuffer::new()));
    let mut buffer = ThreadBuffer::<Task, 2, 2>::new(0, global);

    buffer.push(task(0, 1));
    buffer.push(task(1, 100));
    buffer.push(task(2, 200));

    assert_eq!(buffer.pop().map(|t| t.sequence), Some(1));
    assert_eq!(buffer.pop().map(|t| t.sequence), Some(100));
    assert_eq!(buffer.pop().map(|t| t.sequence), Some(200));
    assert_eq!(buffer.pop(), None);
}

#[test]
fn same_stream_tasks_keep_sequence_when_overflowing_to_global() {
    let global = Arc::new(spin::Mutex::new(GlobalBuffer::new()));
    let mut buffer = ThreadBuffer::<Task, 1, 1>::new(0, global);

    buffer.push(task(42, 1));
    buffer.push(task(42, 2));

    assert_eq!(buffer.pop().map(|t| t.sequence), Some(1));
    assert_eq!(buffer.pop().map(|t| t.sequence), Some(2));
    assert_eq!(buffer.pop(), None);
}

#[test]
fn one_thread_buffer_per_thread_realistic_scenario() {
    let buffers = make_thread_buffers::<2, 2>(3);

    buffers[0].lock().push(task(10, 1));
    buffers[0].lock().push(task(10, 2));
    buffers[1].lock().push(task(20, 3));
    buffers[2].lock().push(task(30, 4));

    assert_eq!(buffers[0].lock().pop().map(|t| t.sequence), Some(1));
    assert_eq!(buffers[0].lock().pop().map(|t| t.sequence), Some(2));
    assert_eq!(buffers[1].lock().pop().map(|t| t.sequence), Some(3));
    assert_eq!(buffers[2].lock().pop().map(|t| t.sequence), Some(4));
}

#[test]
fn one_thread_buffer_per_thread_exceeding_capacity_overflows_to_global() {
    let buffers = make_thread_buffers::<2, 2>(3);

    // Third stream exceeds local stream capacity and must spill to global.
    buffers[0].lock().push(task(100, 1));
    buffers[0].lock().push(task(101, 2));
    buffers[0].lock().push(task(102, 3));

    assert_eq!(buffers[0].lock().pop().map(|t| t.sequence), Some(1));
    assert_eq!(buffers[0].lock().pop().map(|t| t.sequence), Some(2));
    assert_eq!(buffers[0].lock().pop().map(|t| t.sequence), Some(3));
    assert_eq!(buffers[0].lock().pop(), None);
}
