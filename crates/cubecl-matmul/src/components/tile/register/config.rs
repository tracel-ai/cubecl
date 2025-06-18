use cubecl_core::client::ComputeClient;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::{Feature, Runtime};

use crate::components::config::MatmulConfig;
use crate::components::tile::TileConfig;
use crate::components::{Ident, MatmulPrecision, MatrixLayout, TileSize, TilingScheme};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_core::frontend::CubePrimitive;

pub enum ProductType {
    /// Needs lhs to be row major and rhs to be col major
    /// If not the case, tile will be transposed
    Inner,
    /// Needs lhs to be col major and rhs to be row major
    /// If not the case, tile will be transposed
    Outer,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register instruction
pub struct RegisterConfig {
    tiling_scheme: TilingScheme,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    pub stage_dynamic_line_size: bool,
    lhs_global_line_size: u32,
    rhs_global_line_size: u32,
    out_global_line_size: u32,
    lhs_stage_line_size: u32,
    rhs_stage_line_size: u32,
}

impl TileConfig for RegisterConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
        match ident.into() {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_stage_line_size,
            Ident::Rhs => self.rhs_stage_line_size,
            Ident::Out => self.out_global_line_size,
        }
    }

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_global_line_size,
            Ident::Rhs => self.rhs_global_line_size,
            Ident::Out => self.out_global_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.tiling_scheme.tile_size
    }
}

impl MatmulConfig for RegisterConfig {}

impl RegisterConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tiling_scheme: TilingScheme,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        stage_dynamic_line_size: bool,
        lhs_global_line_size: u32,
        rhs_global_line_size: u32,
        out_global_line_size: u32,
        lhs_stage_line_size: u32,
        rhs_stage_line_size: u32,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            tiling_scheme,
            plane_dim,
            lhs_layout,
            rhs_layout,
            stage_dynamic_line_size,
            lhs_global_line_size,
            rhs_global_line_size,
            out_global_line_size,
            lhs_stage_line_size,
            rhs_stage_line_size,
        }
        .validate()?
        .check_availability::<MP, R>(client)
    }

    pub fn product_type(&self) -> ProductType {
        // Best algorithm benchmarked on metal
        // Very surprising that RowCol is better in Outer while
        // ColRow is better in Inner
        match (
            self.matrix_layout(Ident::Lhs),
            self.matrix_layout(Ident::Rhs),
        ) {
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor) => ProductType::Inner,
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor) => ProductType::Outer,
            (MatrixLayout::ColMajor, MatrixLayout::RowMajor) => ProductType::Outer,
            (MatrixLayout::ColMajor, MatrixLayout::ColMajor) => ProductType::Outer,
        }
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        let m = self.tile_size().m();
        let n = self.tile_size().n();
        let k = self.tile_size().k();

        // 128 a bit arbitrary, but accepts 4x4x4 and rejects 8x8x8
        if m * n * k > 128 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile size m-n-k={:?}-{:?}-{:?} is too large for register matmul",
                m, n, k
            ))));
        }

        let lhs = self.stage_line_size(Ident::Lhs);
        let rhs = self.stage_line_size(Ident::Rhs);
        let out = self.global_line_size(Ident::Out);

        match self.matrix_layout(Ident::Lhs) {
            MatrixLayout::RowMajor => {
                if k % lhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                        k, lhs
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if m % lhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                        m, lhs
                    ))));
                }
            }
        }
        match self.matrix_layout(Ident::Rhs) {
            MatrixLayout::RowMajor => {
                if n % rhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                        n, rhs
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if k % rhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                        k, rhs
                    ))));
                }
            }
        }

        if n % out != 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                n, out
            ))));
        }

        Ok(self)
    }

    fn check_availability<MP: MatmulPrecision, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        if self.stage_dynamic_line_size
            && !client
                .properties()
                .feature_enabled(Feature::DynamicLineSize)
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::DynamicLineSizeUnavailable,
            ));
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

        if !(MP::ES::is_supported(client) && MP::EA::is_supported(client)) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TypesUnavailable {
                    input: es,
                    output: ea,
                },
            ));
        }

        Ok(self)
    }
}
