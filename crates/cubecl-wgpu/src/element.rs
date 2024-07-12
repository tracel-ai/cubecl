use crate::compiler::wgsl;

/// The base element trait for the wgpu backend.
pub trait WgpuElement {
    fn wgpu_elem() -> wgsl::Elem;
}

/// The float element type for the wgpu backend.
pub trait FloatElement: WgpuElement {}

/// The int element type for the wgpu backend.
pub trait IntElement: WgpuElement {}

impl WgpuElement for u32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::U32
    }
}

impl WgpuElement for i32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::I32
    }
}

impl WgpuElement for f32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::F32
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
