#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct AmdDevice {
    pub index: usize,
}

impl core::fmt::Debug for AmdDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("AmdDevice({})", self.index))
    }
}
