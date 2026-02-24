mod base;

pub use base::*;

use crate::device::DeviceService;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(not(feature = "std"))]
type Inner<S> = mutex::MutexDeviceHandle<S>;

#[cfg(feature = "std")]
// type Inner<S> = mutex::MutexDeviceHandle<S>;
// type Inner<S> = channel::ChannelDeviceHandle<S>;
type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;

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
    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ()> {
        Ok(Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::insert(device_id, service)?,
        })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::new(device_id),
        }
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
}

// #[cfg(test)]
// mod tests_mutex {
//     type DeviceHandle<S> = mutex::MutexDeviceHandle<S>;
//
//     include!("./tests.rs");
// }

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = reentrant::ReentrantMutexDeviceHandle<S>;

    include!("./tests.rs");
}
