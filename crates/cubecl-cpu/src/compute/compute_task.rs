use crate::{
    compiler::jit::{data::PlironData, engine::PlironEngine},
    compute::notification::Notifications,
};

pub struct ComputeTask {
    pub pliron_engine: PlironEngine,
    pub pliron_data: PlironData,
    pub notifications: Notifications,
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.pliron_engine.run_kernel(&mut self.pliron_data);
        self.pliron_data.complete_unit();
        self.notifications.send();
    }
}
