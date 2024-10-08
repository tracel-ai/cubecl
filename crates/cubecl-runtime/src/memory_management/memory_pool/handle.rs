use crate::id::HandleRef;
use crate::memory_id_type;
use crate::memory_management::MemoryHandle;

// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

impl MemoryHandle<SliceBinding> for SliceHandle {
    fn can_mut(&self) -> bool {
        HandleRef::can_mut(self)
    }

    fn binding(self) -> SliceBinding {
        self.binding()
    }
}
