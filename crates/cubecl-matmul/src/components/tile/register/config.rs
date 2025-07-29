use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_core::ir::{Elem, FloatKind};

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{MatmulPrecision, MatrixLayout, TileIdent, TileSize, TilingScheme};
use cubecl_core::frontend::CubePrimitive;

/// Execution mode for the RegisterMatmul
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during fill
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during fill
    Outer,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register Matmul
pub struct RegisterConfig {
    tiling_scheme: TilingScheme,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
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

    fn matrix_layout(&self, ident: TileIdent) -> MatrixLayout {
        match ident {
            TileIdent::Lhs => self.lhs_layout,
            TileIdent::Rhs => self.rhs_layout,
            TileIdent::Acc => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size(&self, ident: TileIdent) -> u32 {
        match ident {
            TileIdent::Lhs => self.lhs_stage_line_size,
            TileIdent::Rhs => self.rhs_stage_line_size,
            TileIdent::Acc => self.out_global_line_size,
        }
    }

    fn global_line_size(&self, ident: TileIdent) -> u32 {
        match ident {
            TileIdent::Lhs => self.lhs_global_line_size,
            TileIdent::Rhs => self.rhs_global_line_size,
            TileIdent::Acc => self.out_global_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.tiling_scheme.tile_size
    }
}

impl RegisterConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for register matmul
    ///
    /// May return an error if:
    /// - Line sizes do not evenly divide tile sizes in the lined axis
    /// - Types are unavailable
    pub fn new<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tiling_scheme: TilingScheme,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
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
        // TODO: Make it configurable.
        ProductType::Outer
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        let m = self.tile_size().m();
        let n = self.tile_size().n();
        let k = self.tile_size().k();

        let lhs = self.lhs_stage_line_size;
        let rhs = self.rhs_stage_line_size;
        let out = self.out_global_line_size;

        match self.matrix_layout(TileIdent::Lhs) {
            MatrixLayout::RowMajor => {
                if k % lhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {k:?} should be divisible by line size {lhs:?}"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if m % lhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {m:?} should be divisible by line size {lhs:?}"
                    ))));
                }
            }
        }
        match self.matrix_layout(TileIdent::Rhs) {
            MatrixLayout::RowMajor => {
                if n % rhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {n:?} should be divisible by line size {rhs:?}"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if k % rhs != 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {k:?} should be divisible by line size {rhs:?}"
                    ))));
                }
            }
        }

        if n % out != 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in lined axis {n:?} should be divisible by line size {out:?}"
            ))));
        }

        Ok(self)
    }

    fn check_availability<MP: MatmulPrecision, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
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
