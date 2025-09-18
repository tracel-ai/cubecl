use cubecl_core::Runtime;
use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::Numeric;
use cubecl_core::{client::ComputeClient, ir::StorageType};
use cubecl_runtime::{Plane, TypeUsage};

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{MatrixLayout, StageIdent, TileSize, TilingScheme};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register Matmul
pub struct PlaneVecMatInnerProductConfig {
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

impl TileConfig for PlaneVecMatInnerProductConfig {
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
            StageIdent::Acc => 1,
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
        &self.tiling_scheme.tile_size
    }
}

impl PlaneVecMatInnerProductConfig {
    pub fn reduce_line_size(&self) -> u32 {
        self.lhs_stage_line_size
    }

    pub fn n(&self) -> u32 {
        self.tile_size().n()
    }
}

impl PlaneVecMatInnerProductConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for register matmul
    ///
    /// May return an error if:
    /// - Line sizes do not evenly divide tile sizes in the lined axis
    /// - Types are unavailable
    pub fn new<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
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
        .check_availability::<Lhs, Rhs, Acc, R>(client)
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        if self.matrix_layout(StageIdent::Lhs) != MatrixLayout::RowMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Row Major layout is supported for Lhs",
            )));
        }

        if self.matrix_layout(StageIdent::Rhs) != MatrixLayout::ColMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Col Major layout is supported for Rhs",
            )));
        }

        let m = self.tile_size().m();
        let n = self.tile_size().n();
        let k = self.tile_size().k();

        let lhs_line = self.lhs_stage_line_size;
        let rhs_line = self.rhs_stage_line_size;
        let out_line = self.out_global_line_size;

        if m != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Only m=1 is supported, got m={m:?}",
            ))));
        }

        if lhs_line != rhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs must have same line size, got lhs={lhs_line:?} and rhs={rhs_line:?}",
            ))));
        }

        if k != self.plane_dim * lhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be equal to plane_dim times line size (of both lhs and rhs), got k={:?}, plane_dim={:?} line_size={:?}",
                k, self.plane_dim, lhs_line
            ))));
        }

        if !n.is_multiple_of(out_line) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "n must be divisible by out line size, got n={n:?}, out_line_size={out_line:?}",
            ))));
        }

        Ok(self)
    }

    fn check_availability<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        if !client.properties().features.plane.contains(Plane::Ops) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneOpsUnavailable,
            ));
        }

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
