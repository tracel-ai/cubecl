use crate::compute::threadpool::global_buffer::GlobalBuffer;

use super::shared::{Task, task};

#[test]
fn push_creates_fifo_for_new_stream_id() {
    let mut buffer = GlobalBuffer::<Task>::new();

    buffer.push(task(1, 10));
    buffer.push(task(2, 20));

    assert_eq!(
        buffer.pop().map(|t| (t.stream_id, t.sequence)),
        Some((1, 10))
    );
    assert_eq!(
        buffer.pop().map(|t| (t.stream_id, t.sequence)),
        Some((2, 20))
    );
    assert_eq!(buffer.pop(), None);
}

#[test]
fn same_stream_tasks_stay_in_order_inside_their_fifo() {
    let mut buffer = GlobalBuffer::<Task>::new();

    buffer.push(task(7, 1));
    buffer.push(task(7, 2));
    buffer.push(task(7, 3));

    assert_eq!(buffer.pop().map(|t| t.sequence), Some(1));
    assert_eq!(buffer.pop().map(|t| t.sequence), Some(2));
    assert_eq!(buffer.pop().map(|t| t.sequence), Some(3));
    assert_eq!(buffer.pop(), None);
}
