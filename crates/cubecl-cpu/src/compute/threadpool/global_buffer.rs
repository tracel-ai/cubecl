use crate::compute::threadpool::{global_stream_fifo::GlobalStreamFifo, thread_buffer::GetId};

pub struct GlobalBuffer<T: GetId> {
    fifos: Vec<GlobalStreamFifo<T>>,
}

impl<T: GetId> GlobalBuffer<T> {
    pub fn push(&mut self, elem: T) {
        for fifo in self.fifos.iter_mut() {
            if fifo.is_same_id(&elem) {
                fifo.push(elem);
                return;
            }
        }
        self.fifos
            .push_mut(GlobalStreamFifo::new(elem.get_id()))
            .push(elem)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.fifos.iter_mut().find_map(GlobalStreamFifo::pop)
    }
}
