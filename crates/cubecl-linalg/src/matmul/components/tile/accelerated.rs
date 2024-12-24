use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    as_cmma_layout, Ident, MatmulConfigFactory, MatmulPrecision, MatmulProblem, MatmulSize,
    MatrixLayout,
};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::{self as cubecl, Feature};
use cubecl_core::{cmma, prelude::*};

pub struct Accelerated;

impl TileMatmulFamily for Accelerated {
    type Matmul<I: Numeric, O: Numeric> = Accelerated;

    fn size(config: &Self::Config) -> MatmulSize {
        config.size.clone()
    }

    fn input(tile_size: MatmulSize) -> Self::Input {
        tile_size
    }

    fn requires_tensor_cores() -> bool {
        true
    }
}

#[cube]
impl<I: Numeric, O: Numeric> TileMatmul<I, O> for Accelerated {
    type Config = Config;
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
        init_lhs(
            config.layout(Ident::Lhs),
            config.size.m,
            config.size.n,
            config.size.k,
        )
    }

    fn init_rhs(#[comptime] config: Config) -> Self::Rhs {
        init_rhs(
            config.layout(Ident::Rhs),
            config.size.m,
            config.size.n,
            config.size.k,
        )
    }

    fn fill_lhs(slice: &Slice<Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        fill_lhs(slice, lhs, config, config.size.m, config.size.k);
    }

    fn fill_rhs(slice: &Slice<Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        fill_rhs(slice, rhs, config, config.size.n, config.size.k);
    }

    fn fill_accumulator(
        slice: &Slice<Line<O>>,
        acc: &mut Self::Accumulator,
        stride: u32,
        #[comptime] config: Config,
    ) {
        fill_accumulator(slice, acc, stride, config);
    }

    fn read_accumulator<C: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Config,
    ) {
        read_accumulator::<O, C>(out, slice, config.size.n);
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        init_output(config.size.m, config.size.n, config.size.k)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(&acc, O::from_int(0));
    }
}

impl MatmulConfigFactory for Accelerated {
    type Input = MatmulSize;
    type Config = Config;

    fn check_config(config: Self::Config) {
        comptime!(check_plane_dim(config.plane_dim()));
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        check_availability::<MP::ES, MP::EG, R>(config.size.m, config.size.n, config.size.k, client)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        make_config(input, problem, cube_dim, cube_count, advanced_config)
    }
}

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
fn fill_accumulator<C: CubePrimitive, O: Numeric>(
    slice: &Slice<C>,
    acc: &mut cmma::Matrix<O>,
    stride: u32,
    #[comptime] config: Config,
) {
    let layout = comptime!(as_cmma_layout(config.layout(Ident::Out)));
    cmma::load_with_layout(acc, slice, stride, layout);
}

#[cube]
fn init_output<O: Numeric>(
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) -> cmma::Matrix<O> {
    unsafe {
        cmma::Matrix::<O>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
        )
    }
}

#[cube]
fn read_accumulator<O: Numeric, C: Numeric>(
    out: &cmma::Matrix<O>,
    slice: &mut SliceMut<Line<C>>,
    #[comptime] n: u32,
) {
    let acc = cmma::cast::<O, C>(out);
    cmma::store(slice, &acc, n, cmma::MatrixLayout::RowMajor);
}

fn check_availability<I: Numeric, O: Numeric, R: Runtime>(
    m: u32,
    n: u32,
    k: u32,
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<(), MatmulAvailabilityError> {
    let i_elem = I::as_elem_native().expect("to be a native type");
    let o_elem = O::as_elem_native().expect("to be a native type");

    let i_elem = match i_elem {
        Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
        _ => i_elem,
    };

    let o_elem = match o_elem {
        Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
        _ => o_elem,
    };

    if !client.properties().feature_enabled(Feature::Cmma {
        a: i_elem,
        b: i_elem,
        c: o_elem,
        m: m as u8,
        k: k as u8,
        n: n as u8,
    }) {
        return Err(MatmulAvailabilityError::CmmaInstructionUnavailable {
            input: i_elem,
            output: o_elem,
            m,
            n,
            k,
        });
    }

    if !(client.properties().feature_enabled(Feature::Type(i_elem))
        && client.properties().feature_enabled(Feature::Type(o_elem)))
    {
        return Err(MatmulAvailabilityError::TypesUnavailable {
            input: i_elem,
            output: o_elem,
        });
    }

    Ok(())
}

fn check_plane_dim(actual_plane_dim: u32) {
    assert_eq!(32, actual_plane_dim, "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
        actual_plane_dim
    );
}

fn make_config(
    input: MatmulSize,
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
        input,
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
    size: MatmulSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl TileConfig for Config {
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
        size: MatmulSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
