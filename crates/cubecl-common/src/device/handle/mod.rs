mod base;

pub use base::*;

use crate::device::{DeviceService, ServerUtilitiesHandle};

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(all(feature = "std", multi_threading))]
type Inner<S> = channel::ChannelDeviceHandle<S>;
// type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;
#[cfg(all(feature = "std", not(multi_threading)))]
type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner<S> = mutex::MutexDeviceHandle<S>;

/// TODO: Docs
pub struct DeviceHandle<S: DeviceService> {
    handle: Inner<S>,
}

impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[allow(missing_docs)]
impl<S: DeviceService> DeviceHandle<S> {
    pub const fn is_blocking() -> bool {
        Inner::<S>::BLOCKING
    }

    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::insert(device_id, service)?,
        })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::new(device_id),
        }
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    pub fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.submit_blocking(task)
    }

    pub fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R {
        self.handle.submit_blocking_scoped(task)
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.handle.submit(task)
    }

    pub fn flush_queue(&self) {
        self.handle.flush_queue();
    }

    pub fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.exclusive(task)
    }

    pub fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.exclusive_scoped(task)
    }
}

#[cfg(test)]
mod tests_channel {
    type DeviceHandle<S> = channel::ChannelDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = mutex::MutexDeviceHandle<S>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = reentrant::ReentrantMutexDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}
