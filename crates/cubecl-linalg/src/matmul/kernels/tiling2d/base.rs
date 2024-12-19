use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeType};

use super::{block_loop::block_loop, config::CubeTiling2dConfig};

/// Most common tile size, the one used in most tests.
pub(crate) const TILE_SIZE: usize = 4;

#[cube(launch_unchecked)]
#[allow(unused_mut)]
pub fn tiling2d_cube_kernel<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    out: &mut Tensor<Line<F>>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let dims = get_dims::<F>(lhs, rhs);
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<F>(lhs, rhs, out, CUBE_POS_Z);
    let shared_memories = make_shared_memories::<F>(config);
    block_loop::<F>(
        lhs,
        rhs,
        out,
        coordinates,
        offsets,
        shared_memories,
        config,
        dims,
    );
}

// pub mod tiling2d_cube_kernel {
//     use super::*;
//     #[allow(unused, clippy::all)]
//     pub fn expand<F: Float>(
//         context: &mut cubecl::prelude::CubeContext,
//         lhs: <Tensor<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
//         rhs: <Tensor<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
//         out: <Tensor<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
//         config: CubeTiling2dConfig,
//     ) -> <() as cubecl::prelude::CubeType>::ExpandType {
//         use cubecl::prelude::IntoRuntime as _;
//         {
//             let dims = {
//                 let _arg_0 = lhs.clone();
//                 let _arg_1 = rhs.clone();
//                 cubecl::frontend::debug_call_expand(context, "get_dims", |context| {
//                     get_dims::expand::<F>(context, _arg_0.into(), _arg_1.into())
//                 })
//             };
//             let coordinates = {
//                 let _arg_0 = CUBE_POS_X::expand(context);
//                 let _arg_1 = CUBE_POS_Y::expand(context);
//                 let _arg_2 = UNIT_POS::expand(context);
//                 let _arg_3 = config.clone();
//                 cubecl::frontend::debug_call_expand(context, "calculate_coordinates", |context| {
//                     calculate_coordinates::expand(
//                         context,
//                         _arg_0.into(),
//                         _arg_1.into(),
//                         _arg_2.into(),
//                         _arg_3.into(),
//                     )
//                 })
//             };
//             let offsets = {
//                 let _arg_0 = lhs.clone();
//                 let _arg_1 = rhs.clone();
//                 let _arg_2 = out.clone();
//                 let _arg_3 = CUBE_POS_Z::expand(context);
//                 cubecl::frontend::debug_call_expand(context, "calculate_batch_offsets", |context| {
//                     calculate_batch_offsets::expand::<F>(
//                         context,
//                         _arg_0.into(),
//                         _arg_1.into(),
//                         _arg_2.into(),
//                         _arg_3.into(),
//                     )
//                 })
//             };
//             let shared_memories = {
//                 let _arg_0 = config.clone();
//                 cubecl::frontend::debug_call_expand(context, "make_shared_memories", |context| {
//                     make_shared_memories::expand::<F>(context, _arg_0.into())
//                 })
//             };
//             {
//                 let _arg_0 = lhs;
//                 let _arg_1 = rhs;
//                 let _arg_2 = out;
//                 let _arg_3 = coordinates;
//                 let _arg_4 = offsets;
//                 let _arg_5 = shared_memories;
//                 let _arg_6 = config.clone();
//                 let _arg_7 = dims;
//                 cubecl::frontend::debug_call_expand(context, "block_loop", |context| {
//                     block_loop::expand::<F>(
//                         context,
//                         _arg_0.into(),
//                         _arg_1.into(),
//                         _arg_2.into(),
//                         _arg_3.into(),
//                         _arg_4.into(),
//                         _arg_5.into(),
//                         _arg_6.into(),
//                         _arg_7.into(),
//                     )
//                 })
//             };
//             ()
//         }
//     }
//     ///tiling2d_cube_kernel Kernel
//     pub struct Tiling2dCubeKernel<F: Float, __R: cubecl::prelude::Runtime> {
//         settings: cubecl::prelude::KernelSettings,
//         lhs: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         rhs: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         out: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         config: CubeTiling2dConfig,
//         __ty: ::core::marker::PhantomData<(__R, F)>,
//     }
//     #[allow(clippy::too_many_arguments)]
//     impl<F: Float, __R: cubecl::prelude::Runtime> Tiling2dCubeKernel<F, __R> {
//         pub fn new(
//             settings: cubecl::prelude::KernelSettings,
//             lhs: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//             rhs: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//             out: <Tensor<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//             config: CubeTiling2dConfig,
//         ) -> Self {
//             Self {
//                 settings: settings.kernel_name({
//                     let shorten = |p: &'static str| {
//                         if let Some((_, last)) = p.rsplit_once("::") {
//                             last
//                         } else {
//                             p
//                         }
//                     };
//                     let mut name = format!("{}", "tiling2d_cube_kernel");
//                     {
//                         let type_name = shorten(core::any::type_name::<F>());
//                         name.push_str(&format!("_{type_name}"));
//                     }
//                     name
//                 }),
//                 lhs,
//                 rhs,
//                 out,
//                 config,
//                 __ty: ::core::marker::PhantomData,
//             }
//         }
//     }
//
//     impl<F: Float, __R: cubecl::prelude::Runtime> cubecl::Kernel for Tiling2dCubeKernel<F, __R> {
//         fn define(&self) -> cubecl::prelude::KernelDefinition {
//             let mut builder = cubecl::prelude::KernelBuilder::with_local_allocator(
//                 <<__R as cubecl::prelude::Runtime>::Compiler as cubecl::Compiler>::local_allocator(
//                 ),
//             );
//             builder
//                 .context
//                 .register_type::<FloatExpand<0>>(F::as_elem_native_unchecked());
//
//             let lhs = <Tensor<Line<FloatExpand<0>>> as cubecl::prelude::LaunchArgExpand>::expand(
//                 &self.lhs,
//                 &mut builder,
//             );
//             let rhs = <Tensor<Line<FloatExpand<0>>> as cubecl::prelude::LaunchArgExpand>::expand(
//                 &self.rhs,
//                 &mut builder,
//             );
//             let out =
//                 <Tensor<Line<FloatExpand<0>>> as cubecl::prelude::LaunchArgExpand>::expand_output(
//                     &self.out,
//                     &mut builder,
//                 );
//             expand::<FloatExpand<0>>(
//                 &mut builder.context,
//                 lhs.clone(),
//                 rhs.clone(),
//                 out.clone(),
//                 self.config.clone(),
//             );
//             builder.build(self.settings.clone())
//         }
//         fn id(&self) -> cubecl::KernelId {
//             let cube_dim = self.settings.cube_dim.clone();
//             cubecl::KernelId::new::<Self>().info((
//                 cube_dim,
//                 self.config.clone(),
//                 self.lhs.clone(),
//                 self.rhs.clone(),
//                 self.out.clone(),
//             ))
//         }
//     }
//     #[allow(clippy::too_many_arguments)]
//     ///Launch the kernel [tiling2d_cube_kernel()] on the given runtime
//     pub unsafe fn launch_unchecked<'kernel, F: Float, __R: cubecl::prelude::Runtime>(
//         __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
//         __cube_count: cubecl::prelude::CubeCount,
//         __cube_dim: cubecl::prelude::CubeDim,
//         lhs: cubecl::RuntimeArg<'kernel, Tensor<Line<F>>, __R>,
//         rhs: cubecl::RuntimeArg<'kernel, Tensor<Line<F>>, __R>,
//         out: cubecl::RuntimeArg<'kernel, Tensor<Line<F>>, __R>,
//         config: CubeTiling2dConfig,
//     ) -> () {
//         use cubecl::frontend::ArgSettings as _;
//         let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
//         let input_arg_0 =
//             <Tensor<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&lhs);
//         let input_arg_1 =
//             <Tensor<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&rhs);
//         let output_arg_0 =
//             <Tensor<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&out);
//         let __kernel = Tiling2dCubeKernel::<F, __R>::new(
//             __settings,
//             input_arg_0,
//             input_arg_1,
//             output_arg_0,
//             config,
//         );
//         let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
//         lhs.register(&mut launcher);
//         rhs.register(&mut launcher);
//         out.register(&mut launcher);
//         launcher.launch_unchecked(__cube_count, __kernel, __client);
//     }
// }

#[derive(CubeType, Copy, Clone)]
/// Information available at runtime only
/// Strides assume contiguous
pub(crate) struct Dimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<F: Float> {
    pub lhs: SharedMemory<Line<F>>,
    pub rhs: SharedMemory<Line<F>>,
}

#[derive(CubeType, Copy, Clone)]
/// Number of elements in previous batches
/// Not divided by vectorization facto
pub(crate) struct BatchOffsets {
    pub lhs: u32,
    pub rhs: u32,
    pub out: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Coordinates {
    pub unit_row: u32,
    pub unit_col: u32,
    pub skip_row: u32,
    pub skip_col: u32,
}

#[cube]
fn get_dims<F: Float>(lhs: &Tensor<Line<F>>, rhs: &Tensor<Line<F>>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - 2;
    let second_dim = rank - 1;
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}

#[cube]
fn calculate_coordinates(
    cube_pos_x: u32,
    cube_pos_y: u32,
    unit_pos: u32,
    #[comptime] config: CubeTiling2dConfig,
) -> Coordinates {
    let block_size_m = config.block_size_m;
    let block_size_n = config.block_size_n;
    let tile_size = config.tile_size;

    let n_units_per_row = ((block_size_n - 1) / tile_size) + 1;

    // Cube offset
    let skip_row = cube_pos_x * block_size_m;
    let skip_col = cube_pos_y * block_size_n;

    // Position of the first element of the unit, relative to the cube
    let unit_row = (unit_pos / n_units_per_row) * tile_size;
    let unit_col = (unit_pos % n_units_per_row) * tile_size;

    Coordinates {
        unit_row,
        unit_col,
        skip_row,
        skip_col,
    }
}

#[cube]
#[allow(unused_mut)]
fn calculate_batch_offsets<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    out: &Tensor<Line<F>>,
    batch_number: u32,
) -> BatchOffsets {
    let rank = out.rank();

    let dim_m = lhs.shape(rank - 2);
    let dim_n = rhs.shape(rank - 1);

    // Batch offset for output
    let mut offset_out = dim_m * dim_n * batch_number;
    let mut offset_lhs = 0;
    let mut offset_rhs = 0;

    // Batch offset for lhs, rhs
    for b in 0..rank - 2 {
        let tmp = offset_out / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    BatchOffsets {
        lhs: offset_lhs,
        rhs: offset_rhs,
        out: offset_out,
    }
}

#[cube]
fn make_shared_memories<F: Float>(#[comptime] config: CubeTiling2dConfig) -> SharedMemories<F> {
    let tile_size = config.tile_size;
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_k;
    let block_size_n = config.block_size_n;

    let lhs = SharedMemory::<F>::new_lined(block_size_k * block_size_m / tile_size, tile_size);
    let rhs = SharedMemory::<F>::new_lined(block_size_k * block_size_n / tile_size, tile_size);

    SharedMemories::<F> { lhs, rhs }
}
