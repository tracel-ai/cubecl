#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct MetalDevice {
    index: usize,
}

impl MetalDevice {
    pub fn index(&self) -> usize {
        self.index
    }
}
