use std::ffi::c_void;
use std::sync::Arc;

/// Data shared by every unit of a launch: the buffer data pointers (indexed by binding
/// position) and the metadata array. These point into server-owned storage that must outlive
/// the launch.
#[derive(Default)]
pub struct SharedData {
    pub buffer_ptrs: Vec<*mut c_void>,
    pub metadata: Vec<u64>,
}

unsafe impl Send for SharedData {}
unsafe impl Sync for SharedData {}

/// Per-unit kernel arguments. `builtins` holds `[cube_count_x, cube_count_y, cube_count_z,
/// unit_pos_x, unit_pos_y, unit_pos_z]`, matching the order the entry-point pass appends them.
#[derive(Clone, Default)]
pub struct PlironData {
    pub shared: Arc<SharedData>,
    pub builtins: [u32; 6],
}

impl PlironData {
    pub fn new(buffer_ptrs: Vec<*mut c_void>, metadata: Vec<u64>, cube_count: [u32; 3]) -> Self {
        Self {
            shared: Arc::new(SharedData {
                buffer_ptrs,
                metadata,
            }),
            builtins: [cube_count[0], cube_count[1], cube_count[2], 0, 0, 0],
        }
    }

    pub fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.builtins[3] = unit_pos[0];
        self.builtins[4] = unit_pos[1];
        self.builtins[5] = unit_pos[2];
    }

    pub(crate) fn complete_unit(&self) {}
}
