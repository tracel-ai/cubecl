use crate::compute::threadpool::global_stream_fifo::GlobalStreamFifo;

use super::shared::{Task, task};

#[test]
fn pop_returns_none_when_empty() {
    let mut fifo = GlobalStreamFifo::<Task>::new(1);
    assert_eq!(fifo.pop(), None);
}

#[test]
fn keeps_fifo_order() {
    let mut fifo = GlobalStreamFifo::<Task>::new(5);

    fifo.push(task(5, 1));
    fifo.push(task(5, 2));

    assert_eq!(fifo.pop().map(|t| t.sequence), Some(1));
    assert_eq!(fifo.pop().map(|t| t.sequence), Some(2));
    assert_eq!(fifo.pop(), None);
}
