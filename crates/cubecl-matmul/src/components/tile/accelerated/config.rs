use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::Numeric;
use cubecl_core::{Feature, Runtime};
use cubecl_core::{client::ComputeClient, ir::StorageType};

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{MatrixLayout, StageIdent, TileSize};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated Matmul
pub struct AcceleratedConfig {
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

impl TileConfig for AcceleratedConfig {
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

impl AcceleratedConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for accelerated matmul
    ///
    /// May return an error if:
    /// - cmma is unavailable
    /// - cmma is unavailable for given types
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
        .check_availability::<Lhs, Rhs, Acc, R>(client)
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

        let ea = match acc {
            StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
                ElemType::Float(FloatKind::F32).into()
            }
            _ => acc,
        };

        let size = self.tile_size();
        if !client.properties().feature_enabled(Feature::Cmma {
            a: lhs,
            b: rhs,
            c: ea,
            m: size.m() as u8,
            k: size.k() as u8,
            n: size.n() as u8,
        }) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: ea,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        if !(Lhs::is_supported(client) && Rhs::is_supported(client) && Acc::is_supported(client)) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TypesUnavailable {
                    lhs,
                    rhs,
                    output: ea,
                },
            ));
        }

        Ok(self)
    }
}
