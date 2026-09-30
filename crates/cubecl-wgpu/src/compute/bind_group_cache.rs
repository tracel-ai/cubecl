use alloc::sync::Arc;
use core::num::NonZeroU64;

use cubecl_core::zspace::SmallVec;
use cubecl_environment::collections::HashMap;
use wgpu::{BindGroup, ComputePipeline};

use crate::WgpuResource;

/// Number of bind groups a stream caches by default.
///
/// Bind groups are small device-side handles, but a cached entry keeps the
/// buffers it references alive (see [`BindGroupKey`]), so the cache is
/// bounded to keep the retained memory negligible next to the main memory
/// pool. Workloads re-launching kernels over stable buffers - the common
/// case for inference and training loops - settle well below this.
const DEFAULT_CAPACITY: usize = 1024;

/// Cache of [`BindGroup`]s keyed by full binding identity.
///
/// Every kernel launch binds the same pipeline to a list of buffer slices,
/// and launching the same kernel over the same slices builds the exact same
/// bind group every time. Creating it is pure overhead: wgpu has to validate
/// the entries, allocate a new object and (in most backends) hand it to the
/// driver. This cache memoizes that creation - a hit returns the existing
/// object, which is indistinguishable from a freshly created one since bind
/// groups are immutable.
///
/// Soundness of the key is what makes the memoization transparent: it
/// captures everything `create_bind_group` receives, i.e. the pipeline
/// (which determines the bind group layout) and, per entry, the buffer,
/// offset and padded size. Equal keys therefore describe equal descriptors,
/// and equal descriptors produce equal bind groups.
#[derive(Debug)]
pub(super) struct BindGroupCache {
    entries: HashMap<BindGroupKey, CacheEntry>,
    /// Monotonic counter of lookups, used to approximate recency. A `u64`
    /// counter incremented once per launch cannot wrap in any realistic
    /// process lifetime; `saturating_sub` keeps the impossible wrap benign.
    tick: u64,
    capacity: usize,
    /// Bind groups created since the start; used by the tests to tell a hit
    /// from a miss.
    created: usize,
}

/// One cached bind group and when it was last needed.
#[derive(Debug)]
struct CacheEntry {
    group: BindGroup,
    last_used: u64,
}

/// Everything that determines the bind group of a launch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    /// The pipeline the group was created for; the uncached path derives
    /// the layout from it too (`pipeline.get_bind_group_layout(0)`).
    ///
    /// The key owns the `Arc`, and the buffer keys below own their buffers:
    /// while an entry lives, the identities it was built from cannot be
    /// reused by a new pipeline or buffer, so a hit can never return a group
    /// that binds something else than what the key says.
    pipeline: Arc<ComputePipeline>,
    /// One entry per bound resource, in binding order.
    bindings: SmallVec<[BufferBindingKey; 8]>,
}

/// Identity of a single buffer binding: which buffer, and which slice of it.
///
/// `wgpu::Buffer` and `wgpu::ComputePipeline` compare by object identity
/// (wgpu implements `Eq`/`Hash` for its handle types that way), so cloned
/// handles of the same GPU object are equal while distinct objects are not.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferBindingKey {
    buffer: wgpu::Buffer,
    offset: u64,
    /// Size after the 4-byte binding padding applied by
    /// [`WgpuResource::as_wgpu_bind_resource`]; `None` means "the rest of
    /// the buffer", exactly as in the descriptor.
    size: Option<NonZeroU64>,
}

impl BindGroupCache {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            tick: 0,
            capacity: capacity.max(1),
            created: 0,
        }
    }

    /// Number of cached bind groups; test introspection.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no bind group is cached; test introspection.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every cached bind group.
    ///
    /// Bind groups keep their buffers alive, so this releases that retained
    /// memory. Only needed when memory is at a premium - the entries rebuild
    /// on demand.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Return the bind group binding `resources` to `pipeline`, creating and
    /// caching it on the first request.
    pub(super) fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        pipeline: &Arc<ComputePipeline>,
        resources: &[WgpuResource],
    ) -> BindGroup {
        self.tick = self.tick.wrapping_add(1);

        let key = BindGroupKey {
            pipeline: pipeline.clone(),
            bindings: resources
                .iter()
                .map(|resource| BufferBindingKey {
                    buffer: resource.buffer.clone(),
                    offset: resource.offset,
                    size: resource.binding_size(),
                })
                .collect(),
        };

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = self.tick;
            return entry.group.clone();
        }

        self.evict_if_full();

        let entries = resources
            .iter()
            .enumerate()
            .map(|(i, resource)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: resource.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();
        let group_layout = pipeline.get_bind_group_layout(0);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &entries,
        });

        self.entries.insert(
            key,
            CacheEntry {
                group: group.clone(),
                last_used: self.tick,
            },
        );
        self.created += 1;

        group
    }

    /// Make room for one new entry once the cache is at capacity.
    ///
    /// Eviction approximates least-recently-used: in one pass, drop every
    /// entry that has not been touched in the last `capacity` lookups (the
    /// bulk of a saturated cache in a workload that churns bindings). If
    /// everything is that hot, remove the single oldest entry instead.
    fn evict_if_full(&mut self) {
        if self.entries.len() < self.capacity {
            return;
        }

        let cutoff = self.tick.saturating_sub(self.capacity as u64);
        self.entries.retain(|_, entry| entry.last_used > cutoff);

        if self.entries.len() >= self.capacity {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                // Deref before cloning: cloning the reference would borrow
                // the entry we are about to remove.
                .map(|(key, _)| (*key).clone());
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, init_setup};
    use wgpu::BufferUsages;

    fn setup() -> wgpu::Device {
        // `init_setup` registers a client for the device, which can only
        // happen once per process: share the setup between the tests.
        static SETUP: std::sync::OnceLock<wgpu::Device> = std::sync::OnceLock::new();
        SETUP
            .get_or_init(|| {
                init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default())
                    .device
            })
            .clone()
    }

    /// A pipeline whose implicit group 0 has two storage bindings, matching
    /// the two resources every test below binds. Bind groups must populate
    /// every binding of the layout, so single-resource keys would be invalid.
    fn pipeline(device: &wgpu::Device) -> Arc<ComputePipeline> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> a: array<f32>;
                 @group(0) @binding(1) var<storage, read_write> b: array<f32>;
                 @compute @workgroup_size(1)
                 fn main() { b[0] = a[0]; }"
                    .into(),
            ),
        });
        Arc::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            }),
        )
    }

    fn buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn resource(buffer: &wgpu::Buffer, offset: u64, size: u64) -> WgpuResource {
        WgpuResource::new(buffer.clone(), None, offset, size)
    }

    /// Two bindings against `first` (varying slice) and one fixed against
    /// `second`, so a single varying argument changes exactly one key entry.
    fn bindings(
        first: &wgpu::Buffer,
        second: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> [WgpuResource; 2] {
        [resource(first, offset, size), resource(second, 256, 512)]
    }

    #[test]
    fn identical_bindings_hit_the_cache() {
        let device = setup();
        let pipeline = pipeline(&device);
        let mut cache = BindGroupCache::default();
        let (first, second) = (buffer(&device, 4096), buffer(&device, 4096));
        let resources = bindings(&first, &second, 0, 512);

        cache.get_or_create(&device, &pipeline, &resources);
        cache.get_or_create(&device, &pipeline, &resources);

        assert_eq!(cache.created, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn offset_size_and_buffer_are_part_of_the_key() {
        let device = setup();
        let pipeline = pipeline(&device);
        let mut cache = BindGroupCache::default();
        let (first, second) = (buffer(&device, 4096), buffer(&device, 4096));

        cache.get_or_create(&device, &pipeline, &bindings(&first, &second, 0, 512));
        // Same buffer, different offset.
        cache.get_or_create(&device, &pipeline, &bindings(&first, &second, 512, 512));
        // Same buffer and offset, different size.
        cache.get_or_create(&device, &pipeline, &bindings(&first, &second, 0, 513));
        // Same slices, different buffer.
        cache.get_or_create(&device, &pipeline, &bindings(&second, &first, 0, 512));

        assert_eq!(cache.created, 4);
    }

    #[test]
    fn pipeline_is_part_of_the_key() {
        let device = setup();
        let first = pipeline(&device);
        let second = pipeline(&device);
        let mut cache = BindGroupCache::default();
        let (a, b) = (buffer(&device, 4096), buffer(&device, 4096));
        let resources = bindings(&a, &b, 0, 512);

        cache.get_or_create(&device, &first, &resources);
        cache.get_or_create(&device, &second, &resources);

        assert_eq!(cache.created, 2);
    }

    #[test]
    fn eviction_keeps_the_cache_bounded() {
        let device = setup();
        let pipeline = pipeline(&device);
        let mut cache = BindGroupCache::with_capacity(4);
        let (first, second) = (buffer(&device, 4096), buffer(&device, 4096));

        // Distinct slices churn the cache well past its capacity. Offsets
        // respect the device's storage-buffer offset alignment.
        for i in 0..32u64 {
            let resources = bindings(&first, &second, i * 32, 32);
            cache.get_or_create(&device, &pipeline, &resources);
        }

        assert!(cache.len() <= 4, "cache holds {} entries", cache.len());
    }
}
