use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::base::Matmul as _;
use crate::matmul::components::tile::Config as TileConfig;
use crate::matmul::components::{
    as_cmma_layout, tile, Ident, MatmulKernel, MatmulProblem, MatrixLayout,
};
use crate::matmul::kernels::matmul::AdvancedConfig;
use cubecl_core::{self as cubecl, Feature};
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};
use std::marker::PhantomData;

/// Implementations are pairs of element types that are allowed for CMMA
pub trait CmmaValid<I: Numeric, O: Numeric> {}

impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

macro_rules! instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name<I: Numeric, O: Numeric> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
        }

        #[cube]
        impl<I: Numeric, O: Numeric> tile::Matmul<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

            type Lhs = cmma::Matrix<I>;
            type Rhs = cmma::Matrix<I>;
            type Accumulator = cmma::Matrix<O>;

            fn execute(
                lhs: &Self::Lhs,
                rhs: &Self::Rhs,
                out: &mut Self::Accumulator,
                #[comptime] _config: Config,
            ) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(#[comptime] config: Config) -> Self::Lhs {
                init_lhs(config.layout(Ident::Lhs), Self::M, Self::N, Self::K)
            }

            fn init_rhs(#[comptime] config: Config) -> Self::Rhs {
                init_rhs(config.layout(Ident::Rhs), Self::M, Self::N, Self::K)
            }

            fn fill_lhs(slice: &Slice<Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
                fill_lhs(slice, lhs, config, Self::M, Self::K);
            }

            fn fill_rhs(slice: &Slice<Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
                fill_rhs(slice, rhs, config, Self::N, Self::K);
            }

            fn read_accumulator<C: Numeric>(
                out: &Self::Accumulator,
                slice: &mut SliceMut<Line<C>>,
                #[comptime] _config: Config,
            ) {
                read_accumulator::<O, C>(out, slice, Self::N);
            }

            fn init_accumulator(#[comptime] _config: Self::Config) -> Self::Accumulator {
                init_output(Self::M, Self::N, Self::K)
            }

            fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
                cmma::fill(&acc, O::from_int(0));
            }
        }

        impl<I: Numeric, O: Numeric> MatmulKernel<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
            type Config = Config;

            fn check_config(config: Self::Config) {
                comptime!(check_plane_dim(config.plane_dim()));
            }

            fn check_availability<R: Runtime>(
                client: &ComputeClient<R::Server, R::Channel>,
            ) -> Result<(), &str> {
                check_availability::<I, O, R>(Self::M, Self::N, Self::K, client)
            }

            fn make_config(
                problem: &MatmulProblem,
                cube_dim: &CubeDim,
                cube_count: &CubeCount,
                advanced_config: &AdvancedConfig,
            ) -> Self::Config {
                make_config(problem, cube_dim, cube_count, advanced_config)
            }
        }
    };
}

instruction!(Accelerated16x16x16, 16, 16, 16);
instruction!(Accelerated32x8x16, 32, 8, 16);
instruction!(Accelerated8x32x16, 8, 32, 16);

#[cube]
fn execute<I: Numeric, O: Numeric>(
    lhs: &cmma::Matrix<I>,
    rhs: &cmma::Matrix<I>,
    out: &mut cmma::Matrix<O>,
) {
    cmma::execute::<I, I, O, O>(lhs, rhs, out, out);
}

#[cube]
fn init_lhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) -> cmma::Matrix<I> {
    unsafe {
        cmma::Matrix::<I>::uninitialized(cmma::MatrixIdent::A, m, n, k, as_cmma_layout(layout))
    }
}

#[cube]
fn init_rhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) -> cmma::Matrix<I> {
    unsafe {
        cmma::Matrix::<I>::uninitialized(cmma::MatrixIdent::B, m, n, k, as_cmma_layout(layout))
    }
}

#[cube]
fn fill_lhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<C>,
    lhs: &mut cmma::Matrix<I>,
    #[comptime] config: Config,
    #[comptime] m: u32,
    #[comptime] k: u32,
) {
    cmma::load(
        lhs,
        slice,
        match config.layout(Ident::Lhs) {
            MatrixLayout::RowMajor => k,
            MatrixLayout::ColMajor => m,
        },
    );
}

#[cube]
fn fill_rhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<C>,
    rhs: &mut cmma::Matrix<I>,
    #[comptime] config: Config,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    cmma::load(
        rhs,
        slice,
        match config.layout(Ident::Rhs) {
            MatrixLayout::RowMajor => n,
            MatrixLayout::ColMajor => k,
        },
    );
}

#[cube]
fn init_output<O: Numeric>(
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) -> cmma::Matrix<O> {
    unsafe {
        let matrix = cmma::Matrix::<O>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill(&matrix, O::from_int(0));

        matrix
    }
}

#[cube]
fn read_accumulator<O: Numeric, C: Numeric>(
    out: &cmma::Matrix<O>,
    slice: &mut SliceMut<Line<C>>,
    #[comptime] n: u32,
) {
    cmma::store(slice, out, n, cmma::MatrixLayout::RowMajor);
}

fn check_availability<I: Numeric, O: Numeric, R: Runtime>(
    m: u32,
    n: u32,
    k: u32,
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<(), &str> {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: I::as_elem(),
        b: I::as_elem(),
        c: O::as_elem(),
        m: m as u8,
        k: k as u8,
        n: n as u8,
    }) {
        return Err("Cmma not supported.");
    }

    if !(client
        .properties()
        .feature_enabled(Feature::Type(I::as_elem()))
        && client
            .properties()
            .feature_enabled(Feature::Type(O::as_elem())))
    {
        return Err("Types not supported.");
    }

    Ok(())
}

fn check_plane_dim(actual_plane_dim: u32) {
    assert_eq!(32, actual_plane_dim, "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
        actual_plane_dim
    );
}

fn make_config(
    problem: &MatmulProblem,
    cube_dim: &CubeDim,
    _cube_count: &CubeCount,
    advanced_config: &AdvancedConfig,
) -> Config {
    let (lhs_tile_layout, lhs_tile_line_size) = match advanced_config.enforced_tile_layout.0 {
        Some(enforced_layout) if enforced_layout != problem.lhs_layout => (enforced_layout, 1),
        _ => (problem.lhs_layout, problem.lhs_line_size),
    };

    let (rhs_tile_layout, rhs_tile_line_size) = match advanced_config.enforced_tile_layout.1 {
        Some(enforced_layout) if enforced_layout != problem.rhs_layout => (enforced_layout, 1),
        _ => (problem.rhs_layout, problem.rhs_line_size),
    };

    Config::new(
        cube_dim.x,
        lhs_tile_layout,
        rhs_tile_layout,
        lhs_tile_line_size as u32,
        rhs_tile_line_size as u32,
        problem.out_line_size as u32,
    )
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated instruction
pub struct Config {
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl tile::Config for Config {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }

    fn line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }
}

impl MatmulConfig for Config {}

impl Config {
    pub fn new(
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
