use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem, MatmulSize,
    MatrixLayout, as_cmma_layout,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::{self as cubecl, Feature};
use cubecl_core::{cmma, prelude::*};

use super::Tile;

pub struct Accelerated;

impl TileMatmulFamily for Accelerated {
    type Matmul<I: Numeric, O: Numeric> = Accelerated;

    fn tile_shape(config: &Self::Config) -> MatmulSize {
        config.size
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
        cmma::execute::<I, I, O, O>(lhs, rhs, out, out);
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        let size = config.size;
        let layout = config.matrix_layout(Ident::Lhs);
        unsafe {
            cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::A, // Check versus Ident
                size.m,
                size.n,
                size.k,
                as_cmma_layout(layout),
            )
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        let size = config.size;
        let layout = config.matrix_layout(Ident::Rhs);
        unsafe {
            cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::B,
                size.m,
                size.n,
                size.k,
                as_cmma_layout(layout),
            )
        }
    }

    fn fill_lhs(tile: &Tile<I>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Lhs, config);
        cmma::load(lhs, &slice, stride);
    }

    fn fill_rhs(tile: &Tile<I>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Rhs, config);
        cmma::load(rhs, &slice, stride);
    }

    fn fill_accumulator(tile: &Tile<O>, acc: &mut Self::Accumulator, #[comptime] config: Config) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(Ident::Out)));
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Out, config);
        cmma::load_with_layout(acc, &slice, stride, layout);
    }

    fn read_accumulator<C: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Config,
    ) {
        let acc = cmma::cast::<O, C>(out);
        cmma::store(slice, &acc, config.size.n, cmma::MatrixLayout::RowMajor);
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.size;
        unsafe {
            cmma::Matrix::<O>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m,
                size.n,
                size.k,
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, O::from_int(0));
    }
}

impl MatmulConfigFactory for Accelerated {
    type Input = MatmulSize;
    type Config = Config;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        if config.plane_dim != 32 {
            return Err(Box::new(
                "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
            ));
        }
        Ok(())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        let i_elem = MP::ES::as_elem_native().expect("to be a native type");
        let o_elem = MP::EG::as_elem_native().expect("to be a native type");

        let i_elem = match i_elem {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => i_elem,
        };

        let o_elem = match o_elem {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => o_elem,
        };

        let size = config.size;
        if !client.properties().feature_enabled(Feature::Cmma {
            a: i_elem,
            b: i_elem,
            c: o_elem,
            m: size.m as u8,
            k: size.k as u8,
            n: size.n as u8,
        }) {
            return Err(MatmulAvailabilityError::CmmaInstructionUnavailable {
                input: i_elem,
                output: o_elem,
                m: size.m,
                n: size.n,
                k: size.k,
            });
        }

        if !(MP::ES::is_supported(client) && MP::EG::is_supported(client)) {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: i_elem,
                output: o_elem,
            });
        }

        Ok(())
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        _cube_count: &CubeCount,
        _quantized: bool,
    ) -> Self::Config {
        Config::new(
            input,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
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

    fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
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

    fn tile_shape(&self) -> &MatmulSize {
        &self.size
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
