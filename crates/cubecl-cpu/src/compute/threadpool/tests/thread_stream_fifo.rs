use crate::compute::threadpool::thread_stream_fifo::ThreadStreamFifo;

use super::shared::{Task, task};

#[test]
fn keeps_fifo_order_with_simple_push_pop() {
    let mut fifo = ThreadStreamFifo::<Task, 4>::new();

    fifo.push(task(3, 1));
    fifo.push(task(3, 2));

    assert_eq!(fifo.pop().map(|t| t.sequence), Some(1));
    assert_eq!(fifo.pop().map(|t| t.sequence), Some(2));
    assert_eq!(fifo.pop(), None);
}

#[test]
fn keeps_fifo_order_after_wraparound_for_same_stream() {
    let mut fifo = ThreadStreamFifo::<Task, 4>::new();

    fifo.push(task(9, 1));
    fifo.push(task(9, 2));
    fifo.push(task(9, 3));

    assert_eq!(fifo.pop().map(|t| t.sequence), Some(1));

    fifo.push(task(9, 4));

    // Same-stream tasks must remain sequential even after the internal head moved.
    assert_eq!(fifo.pop().map(|t| t.sequence), Some(2));
    assert_eq!(fifo.pop().map(|t| t.sequence), Some(3));
    assert_eq!(fifo.pop().map(|t| t.sequence), Some(4));
    assert_eq!(fifo.pop(), None);
}

#[test]
fn drain_moves_all_items_in_order() {
    let mut source = ThreadStreamFifo::<Task, 4>::new();
    let mut target = ThreadStreamFifo::<Task, 4>::new();

    source.push(task(1, 10));
    source.push(task(1, 11));

    source.drain(&mut target);

    assert!(source.is_empty());
    assert_eq!(target.pop().map(|t| t.sequence), Some(10));
    assert_eq!(target.pop().map(|t| t.sequence), Some(11));
    assert_eq!(target.pop(), None);
}
