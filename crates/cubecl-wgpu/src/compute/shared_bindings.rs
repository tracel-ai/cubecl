use cubecl_runtime::memory_management::SharedMemoryBindings;
use std::sync::{Arc, Mutex};

/// Pool of reusable [`SharedMemoryBindings`] buffers.
///
/// Take a buffer with [`Self::acquire`]. Once the guard drops, the buffer's
/// allocation returns to the free-list.
#[derive(Clone, Default, Debug)]
pub struct SharedBindingsPool {
    free: Arc<Mutex<Vec<SharedMemoryBindings>>>,
}

impl SharedBindingsPool {
    /// Pre-reserve room for `capacity` pooled buffers.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            free: Arc::new(Mutex::new(Vec::with_capacity(capacity))),
        }
    }

    /// Take a buffer, reusing a pooled allocation when available.
    pub fn acquire(&self) -> SharedBindingsGuard {
        let inner = self.free.lock().unwrap().pop().unwrap_or_default();
        SharedBindingsGuard {
            inner,
            pool: self.free.clone(),
        }
    }
}

/// RAII handle to a pooled [`SharedMemoryBindings`] buffer.
///
/// Derefs to the underlying [`SharedMemoryBindings`] so it can be filled and drained in place.
/// On drop it returns the (cleared) buffer to its pool for reuse.
#[derive(Debug)]
pub struct SharedBindingsGuard {
    inner: SharedMemoryBindings,
    pool: Arc<Mutex<Vec<SharedMemoryBindings>>>,
}

impl core::ops::Deref for SharedBindingsGuard {
    type Target = SharedMemoryBindings;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl core::ops::DerefMut for SharedBindingsGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Drop for SharedBindingsGuard {
    fn drop(&mut self) {
        let mut inner = core::mem::take(&mut self.inner);
        // Only pool buffers that actually carry an allocation; `clear` keeps the capacity but
        // drops any bindings not already drained (e.g. a launch that failed before enqueue).
        if inner.bindings.capacity() > 0 {
            inner.clear();
            self.pool.lock().unwrap().push(inner);
        }
    }
}
