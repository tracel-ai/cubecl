use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use crate::cpu::synchronization::SYNC_CUBE_STATE_LEN;

/// Resources shared by all units of a launch.
#[derive(Default)]
pub struct SharedData {
    pub buffer_ptrs: Vec<*mut c_void>,
    pub metadata: Vec<u64>,
    /// Shared barrier counters, initialized to zero.
    pub sync_cube_state: [AtomicU32; SYNC_CUBE_STATE_LEN],
    /// Keeps buffer storage alive until the launch completes.
    pub keepalive: Vec<Box<dyn std::any::Any + Send>>,
}

/// SAFETY: Buffer and shared memory storage outlive the launch. Barrier counters are
/// atomic, and `keepalive` contents are only dropped, never accessed concurrently.
unsafe impl Send for SharedData {}
unsafe impl Sync for SharedData {}

/// Per-unit kernel arguments. Builtin order: cube count x/y/z, then unit position x/y/z.
#[derive(Clone, Default)]
pub struct PlironData {
    pub shared: Arc<SharedData>,
    pub builtins: [u32; 6],
}

impl PlironData {
    pub fn new(
        buffer_ptrs: Vec<*mut c_void>,
        metadata: Vec<u64>,
        cube_count: [u32; 3],
        keepalive: Vec<Box<dyn std::any::Any + Send>>,
    ) -> Self {
        Self {
            shared: Arc::new(SharedData {
                buffer_ptrs,
                metadata,
                sync_cube_state: Default::default(),
                keepalive,
            }),
            builtins: [cube_count[0], cube_count[1], cube_count[2], 0, 0, 0],
        }
    }

    pub fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.builtins[3] = unit_pos[0];
        self.builtins[4] = unit_pos[1];
        self.builtins[5] = unit_pos[2];
    }
}
