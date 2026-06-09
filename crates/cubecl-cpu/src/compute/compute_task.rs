use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::notification::Notifications,
};

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub mlir_data: MlirData,
    pub notifications: Notifications,
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.mlir_data.push_builtin();
        unsafe {
            self.mlir_engine.run_kernel(&mut self.mlir_data);
        }
        self.mlir_data.complete_unit();
        self.notifications.send();
    }
}
