use alloc::vec;
use alloc::vec::Vec;

/// Stages device pointers on the host for async GPU kernel launches.
///
/// When a kernel is launched asynchronously, the GPU runtime (CUDA / HIP) receives a
/// `void**` (pointer-to-pointer) for each argument. The runtime does **not** copy the
/// pointed-to device address eagerly at launch time — it dereferences the host pointer
/// later, when the command processor actually dispatches the kernel from the async queue.
/// This means the host memory backing each staged pointer must remain valid and unchanged
/// until the kernel is dispatched.
///
/// Internally this is a fixed-capacity ring buffer sized so that the oldest slot is never
/// overwritten while the kernel that references it could still be in the async queue.
/// See [`DevicePtrStaging::new`] for the sizing strategy.
pub struct DevicePtrStaging {
    slots: Vec<u64>,
    cursor: usize,
}

impl DevicePtrStaging {
    /// Creates a new staging area.
    ///
    /// # Arguments
    ///
    /// * `max_bindings_per_kernel` — the maximum number of binding slots a single kernel
    ///   launch can consume.
    /// * `max_queue_depth` — the maximum number of kernels that can be in-flight in the
    ///   async queue before a flush occurs.
    ///
    /// The total capacity is `max_bindings_per_kernel * max_queue_depth`, which guarantees
    /// that no slot is recycled while the kernel that references it is still pending.
    pub fn new(max_bindings_per_kernel: usize, max_queue_depth: usize) -> Self {
        Self {
            slots: vec![0; max_bindings_per_kernel * max_queue_depth],
            cursor: 0,
        }
    }

    /// Stages a device pointer and returns a stable host-side reference to it.
    ///
    /// The returned `&u64` points into the internal buffer. The caller passes this
    /// host-side address to the GPU runtime as a `void**` argument, so the slot must
    /// not be overwritten until the kernel has been dispatched.
    pub fn stage(&mut self, device_ptr: u64) -> &u64 {
        self.slots[self.cursor] = device_ptr;
        let host_ref = &self.slots[self.cursor];

        self.cursor += 1;
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        host_ref
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_returns_correct_value() {
        let mut staging = DevicePtrStaging::new(4, 2);
        let r = staging.stage(42);
        assert_eq!(*r, 42);
    }

    #[test]
    fn stage_wraps_around() {
        let max_bindings = 2;
        let max_depth = 2;
        let capacity = max_bindings * max_depth; // 4 slots
        let mut staging = DevicePtrStaging::new(max_bindings, max_depth);

        // Fill all slots.
        for i in 0..capacity {
            staging.stage(i as u64);
        }
        assert_eq!(staging.cursor, 0, "cursor should wrap to 0");

        // Next stage overwrites slot 0.
        let r = staging.stage(99);
        assert_eq!(*r, 99);
        assert_eq!(staging.cursor, 1);
    }

    #[test]
    fn references_remain_stable_within_capacity() {
        let max_bindings = 2;
        let max_depth = 3;
        let capacity = max_bindings * max_depth; // 6 slots
        let mut staging = DevicePtrStaging::new(max_bindings, max_depth);

        // Collect raw pointers for every slot in one full pass.
        let mut ptrs = Vec::new();
        for i in 0..capacity {
            let r = staging.stage(i as u64);
            ptrs.push(r as *const u64);
        }

        // All pointers should still point to distinct, valid addresses with
        // their original values — none have been overwritten yet.
        for (i, &p) in ptrs.iter().enumerate() {
            // SAFETY: the staging area is still alive and no slot has been recycled.
            assert_eq!(unsafe { *p }, i as u64);
        }

        // All pointers should be distinct (no aliasing).
        let unique: std::collections::HashSet<usize> = ptrs.iter().map(|p| *p as usize).collect();
        assert_eq!(unique.len(), capacity);
    }
}
