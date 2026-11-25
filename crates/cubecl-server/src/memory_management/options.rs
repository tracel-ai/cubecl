use super::MemoryAllocationMode;
use alloc::string::String;

/// How a [`MemoryManagement`](super::MemoryManagement) is built: the name its
/// logs give it, and the allocation mode it starts in.
#[derive(Debug)]
pub struct MemoryManagementOptions {
    name: String,
    memory: MemoryAllocationOption,
}

impl MemoryManagementOptions {
    /// Options for a memory named `name`, starting in the mode the runtime
    /// config asks for.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    /// Start in `mode` whatever the runtime config asks for.
    pub fn mode(mut self, mode: MemoryAllocationMode) -> Self {
        self.memory = MemoryAllocationOption::Provided(mode);
        self
    }
}

impl MemoryManagementOptions {
    /// The name and the allocation mode the options hold.
    pub(crate) fn into_parts(self) -> (String, MemoryAllocationOption) {
        (self.name, self.memory)
    }
}

/// Where the [`MemoryAllocationMode`] a memory starts in comes from.
#[derive(Default, Debug)]
pub(crate) enum MemoryAllocationOption {
    /// The runtime config's persistent-memory setting.
    #[default]
    FromConfig,
    /// The mode given.
    Provided(MemoryAllocationMode),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mode given replaces the config's.
    #[test]
    fn a_mode_given_replaces_the_configs() {
        let (name, memory) = MemoryManagementOptions::new("stream")
            .mode(MemoryAllocationMode::Persistent)
            .into_parts();

        assert_eq!(name, "stream");
        assert!(matches!(
            memory,
            MemoryAllocationOption::Provided(MemoryAllocationMode::Persistent)
        ));
    }
}
