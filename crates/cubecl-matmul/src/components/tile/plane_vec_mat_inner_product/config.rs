use cubecl_core::client::ComputeClient;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::Numeric;
use cubecl_core::{Feature, Runtime};

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
    sum_precision: SumPrecision,
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Specifies the precision used for the sum reduction.
pub enum SumPrecision {
    /// Performs the sum in the accumulator precision.
    /// Registers are cast to the accumulator type if needed before summation.
    Accumulator,
    /// Performs the sum in half-precision (f16).
    /// Registers are cast to `f16` if needed before summation.
    F16,
}

impl PlaneVecMatInnerProductConfig {
    pub fn reduce_line_size(&self) -> u32 {
        self.lhs_stage_line_size
    }

    pub fn n(&self) -> u32 {
        self.tile_size().n()
    }

    pub fn sum_precision(&self) -> SumPrecision {
        self.sum_precision
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
        sum_precision: SumPrecision,
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
            sum_precision,
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
                "Only m=1 is supported, got m={:?}",
                m
            ))));
        }

        if lhs_line != rhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs must have same line size, got lhs={:?} and rhs={:?}",
                lhs_line, rhs_line
            ))));
        }

        if k != self.plane_dim * lhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be equal to plane_dim times line size (of both lhs and rhs), got k={:?}, plane_dim={:?} line_size={:?}",
                k, self.plane_dim, lhs_line
            ))));
        }

        if n % out_line != 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "n must be divisible by out line size, got n={:?}, out_line_size={:?}",
                n, out_line
            ))));
        }

        Ok(self)
    }

    fn check_availability<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        if !client.properties().feature_enabled(Feature::PlaneOps) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneOpsUnavailable,
            ));
        }

        let lhs = Lhs::as_elem_native_unchecked();
        let rhs = Rhs::as_elem_native_unchecked();
        let acc = Acc::as_elem_native_unchecked();

        let lhs = match lhs {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => lhs,
        };
        let rhs = match rhs {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => rhs,
        };

        let output = match acc {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => acc,
        };

        if !(Lhs::is_supported(client) && Rhs::is_supported(client) && Acc::is_supported(client)) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
            ));
        }

        if let SumPrecision::F16 = self.sum_precision
            && !client
                .properties()
                .feature_enabled(Feature::Type(Elem::Float(FloatKind::F16)))
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                " Sum Precision is f16 but f16 is not available.  ",
            )));
        }

        Ok(self)
    }
}
