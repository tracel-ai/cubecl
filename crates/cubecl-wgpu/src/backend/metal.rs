use cubecl_core::{
    WgpuCompilationOptions,
    ir::{AddressType, UIntKind},
    prelude::Visibility,
    server::KernelArguments,
};
use cubecl_cpp::{
    metal::{arch::MetalArchitecture, supported_cmma_combinations_metal},
    shared::{MslComputeKernel, register_wmma_features},
};
use cubecl_ir::{
    DeviceProperties, Type,
    features::{AtomicUsage, Plane, TypeUsage},
};
use wgpu::{
    DeviceDescriptor, Features, Limits,
    hal::{self, Adapter, metal},
};

pub fn bindings(repr: &MslComputeKernel, args: &KernelArguments) -> (Vec<Visibility>, usize) {
    let buffers = repr.buffers.iter().map(|it| {
        // When slices are shared, it needs to be read-write if ANY of the slices is read-write,
        // and since we can't be sure, we'll assume everything is read-write.
        if cubecl_server::memory_management::EXCLUSIVE_MEMORY_ONLY {
            *it
        } else {
            Visibility::ReadWrite
        }
    });
    let uniform = args.info.dynamic_metadata_offset >= args.info.data.len();
    let info_vis = (!args.info.data.is_empty()).then_some(match uniform {
        true => Visibility::Uniform,
        false => Visibility::Read,
    });
    (buffers.chain(info_vis).collect(), 0)
}

pub async fn request_metal_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    let features = adapter
        .features()
        .difference(Features::MAPPABLE_PRIMARY_BUFFERS);
    unsafe {
        let hal_adapter = adapter.as_hal::<hal::api::Metal>().unwrap();
        request_device(adapter, &hal_adapter, features, limits)
    }
}

fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &metal::Adapter,
    features: Features,
    limits: Limits,
) -> (wgpu::Device, wgpu::Queue) {
    // The default is MemoryHints::Performance, which tries to do some bigger
    // block allocations. However, we already batch allocations, so we
    // can use MemoryHints::MemoryUsage to lower memory usage.
    let memory_hints = wgpu::MemoryHints::MemoryUsage;
    let device = unsafe {
        adapter
            .open(features, &limits, &memory_hints)
            .expect("should create metal HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
        // SAFETY: Enabling experimental passthrough shaders.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    };

    unsafe {
        wgpu_adapter
            .create_device_from_hal(device, &descriptor)
            .expect("Failed to create wgpu device")
    }
}

#[cfg(target_vendor = "apple")]
pub fn register_metal_features(
    adapter: &wgpu::Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) -> bool {
    // Canary for ensuring the current Metal language version is 3.2+.
    const LAMBDA_CANARY: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void __canary__(device float* out [[buffer(0)]]) {
            auto f = [](float x) -> float { return x * 2.0; };
            out[0] = f(1.0);
        }
    "#;

    let features = adapter.features();
    unsafe {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion};

        let Some(adapter) = adapter.as_hal::<hal::api::Metal>() else {
            return false;
        };
        let raw = adapter.raw_device();

        // Metal's capacity is the working set it recommends staying under.
        // Read before the family and compiler checks below, because either can
        // decline the backend and leave the device on WGSL, where the capacity
        // still holds.
        props
            .memory
            .set_max_memory(raw.recommendedMaxWorkingSetSize());

        // The native feature profile includes plane and CMMA operations that rely on Metal's
        // SIMD-scoped capabilities. Metal can report those capabilities through overlapping
        // programming-model and hardware families:
        // - `Metal3`: the cross-platform programming-model family
        // - `Apple7`: the A14/M1-or-newer hardware family
        // - `Mac2`: the equivalent Mac hardware family, including Apple's paravirtualized device
        //
        // This matches wgpu-hal's SIMD-scoped capability check:
        // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/metal/adapter.rs#L1073-L1077
        //
        // GPU-family support is independent of the supported MSL language version; the canary
        // below verifies MSL 3.2 compiler support separately.
        let has_required_family = raw.supportsFamily(MTLGPUFamily::Metal3)
            || raw.supportsFamily(MTLGPUFamily::Apple7)
            || raw.supportsFamily(MTLGPUFamily::Mac2);

        let support = MslSupport::new(has_required_family, || {
            autoreleasepool(|_| {
                let canary = NSString::from_str(LAMBDA_CANARY);
                let options = MTLCompileOptions::new();
                options.setLanguageVersion(MTLLanguageVersion::Version3_2);
                raw.newLibraryWithSource_options_error(&canary, Some(&options))
                    .map(|_| ())
                    .map_err(|err| format!("{err:?}"))
            })
        });

        match support {
            MslSupport::Supported => {
                comp_options.supports_msl_compiler = true;
                register_features(&adapter, props, features, comp_options);
                true
            }
            MslSupport::FamilyTooOld => {
                log::warn!(
                    "The device predates the Metal 3 GPU family native MSL needs; it runs WGSL instead."
                );
                false
            }
            MslSupport::CompilerTooOld { reason } => {
                // This is a fixable issue, so we should warn users.
                log::warn!(
                    "Device can support native MSL, but Metal compiler version is too old. Upgrading to 3.2 or higher is recommended. MSL 3.2 canary compilation failed: {reason}"
                );
                false
            }
        }
    }
}

/// Whether the device can run this backend's MSL, or why it stays on WGSL.
#[derive(Debug, PartialEq, Eq)]
enum MslSupport {
    Supported,
    /// The device has none of the GPU families carrying the SIMD-scoped operations the plane
    /// and CMMA features rely on.
    FamilyTooOld,
    /// The Metal compiler rejects MSL 3.2, the version the emitted source is written in.
    CompilerTooOld {
        reason: String,
    },
}

impl MslSupport {
    /// `compile_canary` compiles a kernel only MSL 3.2 accepts, and runs only on a device of the
    /// required family.
    fn new(has_required_family: bool, compile_canary: impl FnOnce() -> Result<(), String>) -> Self {
        if !has_required_family {
            return Self::FamilyTooOld;
        }
        match compile_canary() {
            Ok(()) => Self::Supported,
            Err(reason) => Self::CompilerTooOld { reason },
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
pub fn register_metal_features(
    _: &wgpu::Adapter,
    _: &mut DeviceProperties,
    _: &mut WgpuCompilationOptions,
) -> bool {
    false
}

fn register_features(
    _adapter: &metal::Adapter,
    props: &mut DeviceProperties,
    _features: Features,
    _comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props);
    register_cmma(props);
    props.features.alignment = true;
    // This backend emits MSL, so the same `threadgroup_barrier(mem_flags::mem_device)` the
    // native Metal runtime gets.
    props.features.device_memory_scope = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
}

fn register_types(props: &mut DeviceProperties) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind};

    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let types = [
        ElemType::Index,
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    for ty in types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    // MSL has no fp8 type: the emitter stores these as `uint8_t` and converts in software,
    // which is enough to hold them in buffers and cast them, not to compute in them.
    for ty in [FloatKind::E4M3, FloatKind::E5M2, FloatKind::UE8M0] {
        props.register_type_usage(
            ElemType::Float(ty),
            TypeUsage::Conversion | TypeUsage::Buffer,
        );
    }

    // MSL's 64-bit atomics exist only on Apple9 and later and only as min and max, so u64
    // is left out: registering it made add, load and store return wrong values.
    for ty in [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)] {
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
    }
    props.register_atomic_type_usage(
        Type::atomic(ElemType::Float(FloatKind::F32)),
        AtomicUsage::Add | AtomicUsage::LoadStore,
    );
}

fn register_cmma(props: &mut DeviceProperties) {
    let combinations = supported_cmma_combinations_metal(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;

    #[test]
    fn a_device_of_the_family_with_a_current_compiler_runs_msl() {
        assert_eq!(MslSupport::new(true, || Ok(())), MslSupport::Supported);
    }

    /// The family decides before the compiler is asked: a device too old for MSL is not
    /// reported as a compiler to upgrade.
    #[test]
    fn a_device_before_the_family_stays_on_wgsl_without_compiling() {
        let support = MslSupport::new(false, || panic!("the canary must not compile"));
        assert_eq!(support, MslSupport::FamilyTooOld);
    }

    #[test]
    fn a_compiler_before_msl_3_2_stays_on_wgsl() {
        let support = MslSupport::new(true, || Err("no lambdas".into()));
        assert_eq!(
            support,
            MslSupport::CompilerTooOld {
                reason: "no lambdas".into()
            }
        );
    }
}
