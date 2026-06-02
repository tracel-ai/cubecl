use crate::compute::threadpool::thread_buffer::GetId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Task {
    pub(crate) stream_id: usize,
    pub(crate) sequence: usize,
}

impl GetId for Task {
    fn get_id(&self) -> usize {
        self.stream_id
    }
}

pub(crate) fn task(stream_id: usize, sequence: usize) -> Task {
    Task {
        stream_id,
        sequence,
    }
}
