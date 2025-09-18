use cubecl_common::device::{Device, DeviceId};
use cubecl_hip_sys::{HIP_SUCCESS, hipGetDeviceCount};
use std::ffi::c_int;

// It is not clear if AMD has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const AMD_MAX_BINDINGS: u32 = 1024;

#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct AmdDevice {
    pub index: usize,
}

impl core::fmt::Debug for AmdDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("AmdDevice({})", self.index))
    }
}

impl Device for AmdDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u32,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        let mut device_count: c_int = 0;
        let result;
        unsafe {
            result = hipGetDeviceCount(&mut device_count);
        }
        if result == HIP_SUCCESS {
            device_count.try_into().unwrap_or(0)
        } else {
            0
        }
    }
}
