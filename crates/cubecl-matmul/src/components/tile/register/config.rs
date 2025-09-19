use cubecl_core::client::ComputeClient;
use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::Numeric;
use cubecl_core::{Runtime, ir::StorageType};
use cubecl_runtime::TypeUsage;

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{MatrixLayout, StageIdent, TileSize};

/// Execution mode for the RegisterMatmul
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during load
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during load
    Outer,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register Matmul
pub struct RegisterConfig {
    tile_size: TileSize,
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

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        match ident {
            StageIdent::Lhs => self.lhs_layout,
            StageIdent::Rhs => self.rhs_layout,
            StageIdent::Acc => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_stage_line_size,
            StageIdent::Rhs => self.rhs_stage_line_size,
            StageIdent::Acc => self.out_global_line_size,
        }
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_global_line_size,
            StageIdent::Rhs => self.rhs_global_line_size,
            StageIdent::Acc => self.out_global_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.tile_size
    }
}

impl RegisterConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for register matmul
    ///
    /// May return an error if:
    /// - Line sizes do not evenly divide tile sizes in the lined axis
    /// - Types are unavailable
    pub fn new<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_size: TileSize,
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
            tile_size,
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
        .check_availability::<Lhs, Rhs, Acc, R>(client)
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

        match self.matrix_layout(StageIdent::Lhs) {
            MatrixLayout::RowMajor => {
                if !k.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {k:?} should be divisible by line size {lhs:?}"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !m.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {m:?} should be divisible by line size {lhs:?}"
                    ))));
                }
            }
        }
        match self.matrix_layout(StageIdent::Rhs) {
            MatrixLayout::RowMajor => {
                if !n.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {n:?} should be divisible by line size {rhs:?}"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !k.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis {k:?} should be divisible by line size {rhs:?}"
                    ))));
                }
            }
        }

        if !n.is_multiple_of(out) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in lined axis {n:?} should be divisible by line size {out:?}"
            ))));
        }

        Ok(self)
    }

    fn check_availability<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        let lhs = Lhs::as_type_native_unchecked();
        let rhs = Rhs::as_type_native_unchecked();
        let acc = Acc::as_type_native_unchecked();

        let lhs = match lhs {
            StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
                ElemType::Float(FloatKind::F32).into()
            }
            _ => lhs,
        };
        let rhs = match rhs {
            StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
                ElemType::Float(FloatKind::F32).into()
            }
            _ => rhs,
        };

        let output = match acc {
            StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
                ElemType::Float(FloatKind::F32).into()
            }
            _ => acc,
        };

        if !(Lhs::supported_uses(client).contains(TypeUsage::Arithmetic)
            && Rhs::supported_uses(client).contains(TypeUsage::Arithmetic)
            && Acc::supported_uses(client).contains(TypeUsage::Arithmetic))
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
            ));
        }

        Ok(self)
    }
}
