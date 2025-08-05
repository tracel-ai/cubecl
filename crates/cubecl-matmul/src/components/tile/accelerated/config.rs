use cubecl_core::client::ComputeClient;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::{Feature, Runtime};

use crate::components::error::{MatmulAvailabilityError, MatmulSetupError};
use crate::components::tile::TileConfig;
use crate::components::{
    LhsR, MatmulElems, MatmulPrecision, MatrixLayout, RhsR, StageIdent, TileSize, TilingScheme,
};
use cubecl_core::frontend::CubePrimitive;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated Matmul
pub struct AcceleratedConfig {
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
        &self.tiling_scheme.tile_size
    }
}

impl AcceleratedConfig {
    #[allow(clippy::too_many_arguments)]
    /// Create a new config for accelerated matmul
    ///
    /// May return an error if:
    /// - cmma is unavailable
    /// - cmma is unavailable for given types
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
        .check_availability::<MP, R>(client)
    }

    fn check_availability<MP: MatmulPrecision, R: Runtime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<Self, MatmulSetupError> {
        let elems = MatmulElems::new::<MP>();

        let lhs = match elems.lhs_register {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => elems.lhs_register,
        };
        let rhs = match elems.rhs_register {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => elems.rhs_register,
        };

        let ea = match elems.acc {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => elems.acc,
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

        if !(LhsR::<MP>::is_supported(client)
            && RhsR::<MP>::is_supported(client)
            && MP::EA::is_supported(client))
        {
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
