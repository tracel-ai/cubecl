//! One kernel launch: the [`Launcher`].

use cubecl::ir::{ElemType, HardwareProperties, OpaqueType};
use cubecl::prelude::*;

use crate::{
    Arg, Axis, Geometry, Partitioning, PartitioningLaunch, Refusal, Space, SpaceLaunch, Unlabelled,
};

/// How many cubes of how many units the launch runs: stated, or read off the levels.
#[derive(Clone, Debug)]
pub enum Grid {
    Stated {
        cube_count: CubeCount,
        cube_dim: CubeDim,
    },
    FromLevels,
}

/// Whether a device of `hardware` spawns `cube_count` cubes of `cube_dim` units, or the
/// [`Refusal`] saying why not.
///
/// Three caps, and a launch clears all of them: the cube's unit product, the cube's per-axis
/// shape, and the grid's per-axis extent. Checking only the product passes a cube legal in size
/// and illegal in shape: a CPU runtime reports its core count on every axis, and a cube of many
/// planes overruns `max_cube_dim.1` long before the product notices.
///
/// A zero axis is no launch rather than a small grid: cubecl's CUDA server returns before
/// dispatching one, so the outputs keep what they held and the call is timed as though it ran.
/// A dynamic count lives on the device, and the kernel reading one owns its bound.
pub fn spawnable(
    hardware: &HardwareProperties,
    cube_dim: CubeDim,
    cube_count: &CubeCount,
) -> Result<(), Refusal> {
    let units = cube_dim.num_elems();
    if units > hardware.max_units_per_cube {
        return Err(Refusal::CubePastDevice {
            units,
            most: hardware.max_units_per_cube,
        });
    }
    if units == 0 {
        return Err(Refusal::NoUnits);
    }
    let (most_x, most_y, most_z) = hardware.max_cube_dim;
    if cube_dim.x > most_x || cube_dim.y > most_y || cube_dim.z > most_z {
        return Err(Refusal::CubeShapePastDevice {
            shape: (cube_dim.x, cube_dim.y, cube_dim.z),
            most: hardware.max_cube_dim,
        });
    }
    if let CubeCount::Static(x, y, z) = *cube_count {
        if x == 0 || y == 0 || z == 0 {
            return Err(Refusal::NoCubes);
        }
        let (most_x, most_y, most_z) = hardware.max_cube_count;
        if x > most_x || y > most_y || z > most_z {
            return Err(Refusal::GridPastDevice {
                count: (x, y, z),
                most: hardware.max_cube_count,
            });
        }
    }
    Ok(())
}

/// One launch: a kernel-form partitioning, the concrete space with real extents, and the grid.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    partitioning: Partitioning,
    concrete: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    /// The axes some tile reaches past the end of over the concrete space, worked out once: every
    /// operand the launch binds asks for them.
    overhanging: Vec<Axis>,
    /// The leaf over the concrete space.
    leaf: Space,
}

impl Launcher {
    /// `partitioning` over `concrete`'s real extents, on `grid`, or the [`Refusal`] saying why the
    /// device cannot run it.
    pub fn new(
        client: &Client,
        partitioning: Partitioning,
        concrete: &Space,
        grid: Grid,
    ) -> Result<Self, Refusal> {
        // The kernel-form space may state no extents.
        let over_concrete = Partitioning::new(concrete.clone(), partitioning.levels().to_vec());
        let (cube_count, cube_dim) = match grid {
            Grid::Stated {
                cube_count,
                cube_dim,
            } => (cube_count, cube_dim),
            Grid::FromLevels => {
                let plane_size = client.properties().hardware.plane_size_max;
                let plane_units = partitioning.units();
                if plane_units != 1 && plane_units != plane_size {
                    return Err(Refusal::GridNotReadOffLevels {
                        units: plane_units,
                        plane: plane_size,
                    });
                }
                (
                    over_concrete.cube_count(),
                    over_concrete.cube_dim(plane_size),
                )
            }
        };
        spawnable(&client.properties().hardware, cube_dim, &cube_count)?;
        let fillers = partitioning.fillers();
        // Checked on the host: a panic at expansion is unseen and the launch returns zeros.
        let barriers = client
            .properties()
            .features
            .types
            .opaque
            .contains(&OpaqueType::Barrier);
        if fillers > 0 && !barriers {
            return Err(Refusal::FillersWithoutBarriers { fillers });
        }
        Ok(Launcher {
            client: client.clone(),
            partitioning,
            concrete: concrete.clone(),
            cube_count,
            cube_dim,
            overhanging: over_concrete.overhanging(),
            leaf: over_concrete.leaf(),
        })
    }

    /// This launch, each cube holding `bytes` of shared memory, or the [`Refusal`] saying the device
    /// gives one less: what the device gives a cube, less what it reserves for cube-scoped matrices.
    pub fn holding(self, bytes: usize) -> Result<Self, Refusal> {
        let hardware = &self.client.properties().hardware;
        let most = hardware
            .max_shared_memory_size
            .saturating_sub(hardware.cube_mma_reserved_shared_memory);
        match bytes > most {
            true => Err(Refusal::SharedMemoryPastDevice { bytes, most }),
            false => Ok(self),
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn cube_count(&self) -> CubeCount {
        self.cube_count.clone()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.cube_dim
    }

    /// The concrete space: this launch's real extents.
    pub fn space(&self) -> &Space {
        &self.concrete
    }

    /// The partitioning this launch states, its space in kernel form.
    pub fn partitioning(&self) -> &Partitioning {
        &self.partitioning
    }

    /// The kernel's partitioning argument, dynamic extents sized by this launch.
    pub fn partitioning_arg(&self) -> PartitioningLaunch {
        PartitioningLaunch::new(self.space_arg(), self.partitioning.levels().to_vec())
    }

    /// The kernel-form space as a kernel argument.
    fn space_arg(&self) -> SpaceLaunch {
        let kernel = self.partitioning.space();
        let mut sizes = SequenceArg::new();
        if !kernel.is_static() {
            for axis in kernel.axes() {
                sizes.push(self.concrete.extent(axis));
            }
        }
        SpaceLaunch::new(kernel.shape().clone(), sizes)
    }

    /// The axes some tile reaches past the end of, whose accesses are masked.
    pub(crate) fn overhangs(&self) -> &[Axis] {
        &self.overhanging
    }

    /// The leaf edge along `axis`: the stated tile, or the whole extent where none was.
    fn leaf_edge(&self, axis: Axis) -> usize {
        self.leaf
            .extents()
            .iter()
            .find(|&&(a, _)| a == axis)
            .map(|&(_, edge)| edge)
            .unwrap_or_else(|| self.concrete.extent(axis))
    }

    /// Bind `binding`, whose buffer holds `element`, as an operand of this launch: for a
    /// [`packed`](Arg::packed) one, the words its fields are stored in.
    pub fn arg(&self, binding: TensorBinding, element: ElemType) -> Arg<'_, Unlabelled> {
        Arg::bound(self, binding, element)
    }

    /// [`arg`](Self::arg) over a stated geometry, for an operand with no tensor; end it with
    /// [`build_spec`](Arg::build_spec).
    pub fn unbound(&self, geometry: &Geometry) -> Arg<'_, Unlabelled> {
        Arg::unbound(self, geometry)
    }

    /// The widest line every operand can be served in along `axis`, which must label each
    /// operand's innermost dim.
    pub fn vector_size(
        &self,
        axis: Axis,
        operands: &[(&Geometry, &[Axis])],
        type_size: usize,
    ) -> usize {
        for (_, axes) in operands {
            assert_eq!(
                axes.last(),
                Some(&axis),
                "Launcher::vector_size: axis {axis:?} must label each operand's innermost dim"
            );
        }
        // A masked access counts in lines and would clip wrongly, so overhangs serve scalar.
        let overhangs = self.overhangs();
        let masked = operands
            .iter()
            .any(|(_, axes)| axes.iter().any(|a| overhangs.contains(a)));
        if masked {
            return 1;
        }
        let leaf = self.leaf_edge(axis);
        self.client
            .io_optimized_vector_sizes(type_size)
            .filter(|&v| {
                leaf.is_multiple_of(v)
                    && operands
                        .iter()
                        .all(|(g, axes)| g.serves(&[(axis, v)], axes).is_ok())
            })
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid with a zero axis is refused, not launched: cubecl's CUDA server returns without
    /// dispatching one, so the launch would answer with an uncomputed buffer. Every axis, since
    /// the count is read off whichever axis a level put there.
    #[test]
    fn a_grid_with_a_zero_axis_is_refused() {
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 30,
            max_shared_memory_size: 32 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: 4,
            cube_mma_reserved_shared_memory: 0,
        };
        let cube_dim = CubeDim::new_2d(32, 1);
        for count in [
            CubeCount::Static(0, 8, 1),
            CubeCount::Static(64, 0, 1),
            CubeCount::Static(64, 8, 0),
        ] {
            assert_eq!(
                spawnable(&hardware, cube_dim, &count),
                Err(Refusal::NoCubes),
                "{count:?}"
            );
        }
        assert_eq!(
            spawnable(&hardware, cube_dim, &CubeCount::Static(64, 8, 1)),
            Ok(())
        );
    }
}
