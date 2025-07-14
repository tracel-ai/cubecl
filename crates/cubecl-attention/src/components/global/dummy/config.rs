use crate::components::global::GlobalConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyGlobalConfig {}

impl GlobalConfig for DummyGlobalConfig {}
