use crate::compute::threadpool::thread_buffer::GetId;

pub struct ThreadStreamFifo<T: GetId, const CAPACITY: usize> {
    buffer: [Option<T>; CAPACITY],
    front: usize,
    len: usize,
    id: usize,
}

impl<T: GetId, const CAPACITY: usize> ThreadStreamFifo<T, CAPACITY> {
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_full(&self) -> bool {
        self.len == CAPACITY
    }

    pub fn is_same_id(&self, elem: &T) -> bool {
        self.id == elem.get_id()
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let elem = self.buffer[self.front].take();
        self.front = (self.front + 1) % CAPACITY;
        self.len -= 1;
        elem
    }

    pub fn push(&mut self, elem: T) {
        if self.is_empty() {
            self.id = elem.get_id();
        }
        debug_assert!(self.is_same_id(&elem));
        debug_assert!(self.len < CAPACITY);
        self.buffer[self.len % CAPACITY] = Some(elem);
        self.len += 1;
    }

    pub fn drain(&mut self, target: &mut Self) {
        for i in self.front..(self.front + self.len) {
            let i = i % CAPACITY;
            target.push(self.buffer[i].take().unwrap());
        }
        self.front = 0;
        self.len = 0;
    }
}
