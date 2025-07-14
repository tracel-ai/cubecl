use crate::components::stage::StageConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig {}

impl StageConfig for DummyStageConfig {}
