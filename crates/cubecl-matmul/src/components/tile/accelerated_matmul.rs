use crate::components::config::MatmulConfig;
use crate::components::resource::ComputeResources;
use crate::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::components::{
    Ident, InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
    MatmulProblem, MatrixLayout, TileSize, as_cmma_layout,
};
use crate::kernels::MatmulAvailabilityError;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::{self as cubecl, Feature};
use cubecl_core::{cmma, prelude::*};

use super::{Tile, TileMatmulConfigInput};

pub struct AcceleratedMatmul;

impl TileMatmulFamily for AcceleratedMatmul {
    type Matmul<MP: MatmulPrecision> = AcceleratedMatmul;

    fn requires_tensor_cores() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }
}

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for AcceleratedMatmul {
    type Config = Config;
    type Lhs = cmma::Matrix<MP::ES>;
    type Rhs = cmma::Matrix<MP::ES>;
    type Accumulator = cmma::Matrix<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] _config: Config,
    ) {
        cmma::execute::<MP::ES, MP::ES, MP::EA, MP::EA>(lhs, rhs, out, out);
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        let size = config.size;
        let layout = config.matrix_layout(Ident::Lhs);
        unsafe {
            cmma::Matrix::<MP::ES>::uninitialized(
                cmma::MatrixIdent::A,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        let size = config.size;
        let layout = config.matrix_layout(Ident::Rhs);
        unsafe {
            cmma::Matrix::<MP::ES>::uninitialized(
                cmma::MatrixIdent::B,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Lhs, config);
        cmma::load(lhs, &slice, stride);
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Rhs, config);
        cmma::load(rhs, &slice, stride);
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(Ident::Out)));
        let (slice, stride) = tile.as_unlined::<Config>(Ident::Out, config);
        cmma::load_with_layout(acc, &slice, stride, layout);
    }

    fn write_results(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Config,
    ) {
        let acc = cmma::cast::<MP::EA, MP::EO>(out);
        cmma::store(slice, &acc, config.size.n(), cmma::MatrixLayout::RowMajor);
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.size;
        unsafe {
            cmma::Matrix::<MP::EA>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, MP::EA::from_int(0));
    }
}

impl MatmulConfigFactory for AcceleratedMatmul {
    type Input = TileMatmulConfigInput;
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
        if config.stage_dynamic_line_size
            && !client
                .properties()
                .feature_enabled(Feature::DynamicLineSize)
        {
            return Err(MatmulAvailabilityError::DynamicLineSizeUnavailable);
        }

        let es = MP::ES::as_elem_native().expect("to be a native type");
        let ea = MP::EA::as_elem_native().expect("to be a native type");

        let es = match es {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => es,
        };

        let ea = match ea {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => ea,
        };

        let size = config.size;
        if !client.properties().feature_enabled(Feature::Cmma {
            a: es,
            b: es,
            c: ea,
            m: size.m() as u8,
            k: size.k() as u8,
            n: size.n() as u8,
        }) {
            return Err(MatmulAvailabilityError::CmmaInstructionUnavailable {
                input: es,
                output: ea,
                size: Some(TileSize::new(size.m(), size.n(), size.k())),
            });
        }

        if !(MP::ES::is_supported(client) && MP::EA::is_supported(client)) {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: es,
                output: ea,
            });
        }

        Ok(())
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        _cube_count: &CubeCount,
        _quantized: bool,
    ) -> Self::Config {
        let (lhs_line_size, rhs_line_size, stage_line_size_update) =
            if input.vectorization.stage_line_size == 0 {
                (line_sizes.lhs as u32, line_sizes.rhs as u32, false)
            } else {
                (
                    input.vectorization.stage_line_size as u32,
                    input.vectorization.stage_line_size as u32,
                    true,
                )
            };
        Config::new(
            input.tile_size,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_line_size,
            rhs_line_size,
            line_sizes.out as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated instruction
pub struct Config {
    size: TileSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    stage_dynamic_line_size: bool,
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

    fn stage_line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.size
    }
}

impl MatmulConfig for Config {}

impl Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        size: TileSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        stage_dynamic_line_size: bool,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            stage_dynamic_line_size,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
