//! What a Vulkan device has left for storage buffers, read through the handles wgpu holds.

use ash::vk;
use wgpu::{BufferUsages, BufferUses, hal};

/// The memory type a storage buffer is bound to and whether it is a unified one: the first
/// type `memory_type_bits` allows whose flags hold `DEVICE_LOCAL`, as wgpu's allocator picks
/// it, or, when `unified` is asked on an integrated GPU, the first one host visible and
/// coherent as well, as `create_storage_buffer` picks it there, where host memory is the
/// device's.
pub(crate) fn storage_memory_type(
    instance: &ash::Instance,
    phys_device: vk::PhysicalDevice,
    memory_type_bits: u32,
    unified: bool,
) -> Option<(usize, bool)> {
    // SAFETY: queries on a physical device wgpu holds open, answered by value.
    let (memory, properties) = unsafe {
        (
            instance.get_physical_device_memory_properties(phys_device),
            instance.get_physical_device_properties(phys_device),
        )
    };
    let find = |flags: vk::MemoryPropertyFlags| {
        memory
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .filter(|(index, _)| memory_type_bits & (1 << *index) != 0)
            .find(|(_, memory_type)| memory_type.property_flags.contains(flags))
            .map(|(index, _)| index)
    };
    // On a discrete GPU, host-visible memory is the BAR window, often only 256 MiB.
    let integrated = unified && properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
    let unified = integrated
        .then(|| {
            find(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .flatten()
        .map(|index| (index, true));
    unified.or_else(|| find(vk::MemoryPropertyFlags::DEVICE_LOCAL).map(|index| (index, false)))
}

/// What the driver's budget leaves on the heap a storage buffer of `usage` is bound to,
/// through the `VK_EXT_memory_budget` wgpu enables when the driver offers it. The heap is the
/// one an allocation picks: a probe buffer, bound to no memory, asks the driver which memory
/// types such a buffer may take, and `vk_storage` says whether `create_storage_buffer` or
/// wgpu does the allocating. `None` on another backend or without the extension.
pub(crate) fn memory_available(
    device: &wgpu::Device,
    usage: BufferUsages,
    vk_storage: bool,
) -> Option<u64> {
    // SAFETY: the raw handles are only read, for queries that change no state.
    let hal = unsafe { device.as_hal::<hal::api::Vulkan>() }?;
    if !hal
        .enabled_device_extensions()
        .contains(&ash::ext::memory_budget::NAME)
    {
        return None;
    }
    let instance = hal.shared_instance().raw_instance();
    let phys_device = hal.raw_physical_device();
    let mut flags = hal::vulkan::conv::map_buffer_usage(map_buffer_usage(usage));
    if vk_storage {
        flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    let info = vk::BufferCreateInfo::default()
        .size(1)
        .usage(flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: a buffer bound to no memory, destroyed once its requirements are read.
    let memory_type_bits = unsafe {
        let raw = hal.raw_device();
        let probe = raw.create_buffer(&info, None).ok()?;
        let requirements = raw.get_buffer_memory_requirements(probe);
        raw.destroy_buffer(probe, None);
        requirements.memory_type_bits
    };
    let (memory_type, _) =
        storage_memory_type(instance, phys_device, memory_type_bits, vk_storage)?;
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: a Vulkan 1.1 query into structures this function owns, on the device's own
    // instance.
    unsafe { instance.get_physical_device_memory_properties2(phys_device, &mut properties) };
    let heap = properties.memory_properties.memory_types[memory_type].heap_index as usize;
    Some(budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap]))
}

pub(crate) fn map_buffer_usage(usage: BufferUsages) -> BufferUses {
    let mut u = BufferUses::empty();
    u.set(BufferUses::MAP_READ, usage.contains(BufferUsages::MAP_READ));
    u.set(
        BufferUses::MAP_WRITE,
        usage.contains(BufferUsages::MAP_WRITE),
    );
    u.set(BufferUses::COPY_SRC, usage.contains(BufferUsages::COPY_SRC));
    u.set(BufferUses::COPY_DST, usage.contains(BufferUsages::COPY_DST));
    u.set(BufferUses::INDEX, usage.contains(BufferUsages::INDEX));
    u.set(BufferUses::VERTEX, usage.contains(BufferUsages::VERTEX));
    u.set(BufferUses::UNIFORM, usage.contains(BufferUsages::UNIFORM));
    u.set(
        BufferUses::STORAGE_READ_ONLY | BufferUses::STORAGE_READ_WRITE,
        usage.contains(BufferUsages::STORAGE),
    );
    u.set(BufferUses::INDIRECT, usage.contains(BufferUsages::INDIRECT));
    u.set(
        BufferUses::QUERY_RESOLVE,
        usage.contains(BufferUsages::QUERY_RESOLVE),
    );
    u.set(
        BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
        usage.contains(BufferUsages::BLAS_INPUT),
    );
    u.set(
        BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
        usage.contains(BufferUsages::TLAS_INPUT),
    );
    u
}
