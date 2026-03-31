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
///
/// # Safety invariant
///
/// The ring buffer capacity **must** equal `max_bindings_per_kernel × max_queue_depth`,
/// where `max_queue_depth` is **twice** [`FlushingPolicy::max_check_count`]. The factor
/// of two comes from the [`PendingDropQueue`]'s double-buffer scheme: on each flush the
/// staged batch rotates to *pending*, but the previous pending batch is only freed at
/// that point. This means kernels from **two** consecutive flush cycles can still have
/// live references into the ring buffer simultaneously. If the capacity were only
/// `1 × max_check_count × max_bindings`, the cursor could wrap and overwrite a slot
/// that a kernel from the prior (still-pending) cycle still references.
///
/// If these values fall out of sync, the GPU runtime may read a stale or overwritten
/// device address, causing memory corruption or use-after-free on the device.
///
/// [`FlushingPolicy::max_check_count`]: crate::memory_management::drop_queue::FlushingPolicy::max_check_count
/// [`PendingDropQueue`]: crate::memory_management::drop_queue::PendingDropQueue
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
    ///   async queue before a flush occurs (i.e. [`FlushingPolicy::max_check_count`]).
    ///
    /// The total capacity is `max_bindings_per_kernel * max_queue_depth * 2`. The `× 2`
    /// accounts for the [`PendingDropQueue`]'s double-buffer: after a flush, kernels from
    /// **two** consecutive cycles (the current *pending* batch and the new *staged* batch)
    /// may still reference slots in this buffer.
    ///
    /// [`FlushingPolicy::max_check_count`]: crate::memory_management::drop_queue::FlushingPolicy::max_check_count
    /// [`PendingDropQueue`]: crate::memory_management::drop_queue::PendingDropQueue
    pub fn new(max_bindings_per_kernel: usize, max_queue_depth: usize) -> Self {
        Self {
            slots: vec![0; max_bindings_per_kernel * max_queue_depth * 2],
            cursor: 0,
        }
    }

    /// Stages a device pointer and returns a stable host-side reference to it.
    ///
    /// The returned `&u64` points into the internal buffer. The caller passes this
    /// host-side address to the GPU runtime as a `void**` argument, so the slot must
    /// not be overwritten until the kernel has been dispatched.
    ///
    /// # Correctness
    ///
    /// Each call to `stage` advances the cursor by one. A single kernel launch may
    /// call `stage` up to `max_bindings_per_kernel` times. The ring buffer is sized
    /// for two full flush cycles (`2 × max_queue_depth` launches), so the caller
    /// **must** issue a fence (via [`PendingDropQueue::flush`]) at least every
    /// `max_queue_depth` launches to guarantee all prior kernels have been dispatched
    /// and no longer reference the about-to-be-recycled slots.
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
        // Actual capacity is max_bindings * max_depth * 2 (double-buffer factor).
        let capacity = max_bindings * max_depth * 2; // 8 slots
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
