// Base dialect

// pub trait Dialect:
//     DialectIncludes<Self>
//     + DialectTypes<Self>
//     + DialectBindings<Self>
//     + DialectCubeBuiltins<Self>
//     + Default
//     + Clone
//     + Copy
//     + Debug
//     + Send
//     + Sync
//     + Eq
//     + Hash
//     + 'static
// {
//     type Architecture: Architecture;
// }

// // Includes

// pub trait DialectIncludes<D: Dialect> {
//     type Extension: Debug + Clone + Sync + Send;

//     fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags<D>) -> std::fmt::Result;
//     fn compile_extensions(
//         f: &mut std::fmt::Formatter<'_>,
//         extensions: &[Self::Extension],
//     ) -> std::fmt::Result;
// }

// // Types

// pub trait DialectTypes<D: Dialect> {
//     fn compile_type_definitions(
//         f: &mut std::fmt::Formatter<'_>,
//         items: &HashSet<Item<D>>,
//         scalars: &[(Elem<D>, usize)],
//         info: &cubecl_core::Info,
//         flags: &Flags,
//     ) -> std::fmt::Result;
//     fn compile_shared_memory_declaration(
//         f: &mut std::fmt::Formatter<'_>,
//         shared: &SmemAllocation,
//     ) -> std::fmt::Result {
//         let SmemAllocation { value, offset, .. } = shared;
//         let ptr = value.id(ctx);
//         let ptr_ty = ptr.item();
//         let size_bytes = shared.size();
//         writeln!(f, "// Shared value size: {size_bytes} bytes")?;
//         writeln!(
//             f,
//             "{ptr_ty} {ptr} = reinterpret_cast<{ptr_ty}>(&dynamic_shared_mem[{offset}]);"
//         )
//     }
//     fn compile_polyfills(_f: &mut std::fmt::Formatter<'_>, _flags: &Flags<D>) -> std::fmt::Result {
//         Ok(())
//     }
// }

// // Kernel argument bindings

// pub trait DialectBindings<D: Dialect> {
//     fn compile_bindings_body(
//         _f: &mut std::fmt::Formatter<'_>,
//         _body: &Body<D>,
//     ) -> std::fmt::Result {
//         Ok(())
//     }
// }

// // Cube builtins dialect

// pub trait DialectCubeBuiltins<D: Dialect> {
//     /// Depending on the dialect available built-ins the
//     /// inclusion rules might change.
//     /// For instance in metal we have a built-in for the Unit plane position
//     /// but in other dialects there is none so we have to compute it using
//     /// other built-ins.
//     fn builtin_rules(flags: &CubeIndexFlags) -> CubeIndexFlags {
//         let unit_pos_plane = flags.unit_pos_plane;
//         let plane_dim = flags.plane_dim || unit_pos_plane;
//         let plane_pos = flags.plane_pos;
//         let absolute_pos = flags.absolute_pos || unit_pos_plane;
//         let absolute_pos_tuple = flags.absolute_pos_tuple || absolute_pos;
//         let cube_dim = flags.cube_dim;
//         let cube_dim_tuple = flags.cube_dim_tuple || cube_dim || absolute_pos;
//         let unit_pos = flags.unit_pos;
//         let unit_pos_tuple = flags.unit_pos_tuple || unit_pos;
//         let cube_count = flags.cube_count;
//         let cube_count_tuple = flags.cube_count_tuple || absolute_pos;
//         let cube_pos = flags.cube_pos;
//         let cube_pos_tuple = flags.cube_pos_tuple || cube_pos;
//         let cluster_group = flags.cluster_pos;

//         CubeIndexFlags {
//             absolute_pos,
//             absolute_pos_tuple,
//             cube_count,
//             cube_count_tuple,
//             cube_dim,
//             cube_dim_tuple,
//             cube_pos,
//             cube_pos_tuple,
//             plane_dim,
//             plane_pos,
//             unit_pos_tuple,
//             unit_pos,
//             unit_pos_plane,
//             cluster_pos: cluster_group,
//         }
//     }

//     fn compile_absolute_pos_tuple_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let value = Builtin::<D>::AbsolutePosBaseName;
//         let ty = value.item();
//         let cube_pos_x = Builtin::<D>::CubePosX;
//         let cube_pos_y = Builtin::<D>::CubePosY;
//         let cube_pos_z = Builtin::<D>::CubePosZ;
//         let cube_dim_x = Builtin::<D>::CubeDimX;
//         let cube_dim_y = Builtin::<D>::CubeDimY;
//         let cube_dim_z = Builtin::<D>::CubeDimZ;
//         let unit_pos_x = Builtin::<D>::UnitPosX;
//         let unit_pos_y = Builtin::<D>::UnitPosY;
//         let unit_pos_z = Builtin::<D>::UnitPosZ;
//         writeln!(
//             f,
//             "{ty} {value} = make_{ty}(
//     {cube_pos_x} * {cube_dim_x} + {unit_pos_x},
//     {cube_pos_y} * {cube_dim_y} + {unit_pos_y},
//     {cube_pos_z} * {cube_dim_z} + {unit_pos_z}
// );"
//         )
//     }

//     fn compile_unit_pos_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let value = Builtin::<D>::UnitPos;
//         let ty = value.item();
//         let cube_dim_x = Builtin::<D>::CubeDimX;
//         let cube_dim_y = Builtin::<D>::CubeDimY;
//         let unit_pos_x = Builtin::<D>::UnitPosX;
//         let unit_pos_y = Builtin::<D>::UnitPosY;
//         let unit_pos_z = Builtin::<D>::UnitPosZ;
//         writeln!(
//             f,
//             "{ty} {value} = {unit_pos_x} + {unit_pos_y} * {cube_dim_x} + {unit_pos_z} * ({cube_dim_x} * {cube_dim_y});"
//         )
//     }
// }
