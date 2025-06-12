use crate::components::ComputeResources;
use crate::components::MatmulChecker;
use crate::components::MatmulProblem;
use crate::components::TilingScheme;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::CommonStageConfig;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::partitioned_stage::unit_partitioned::UnitMatmul;
use crate::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::TileSetupInput;
use crate::components::{InvalidConfigError, MatmulLineSizes, MatmulPrecision};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::StageInput;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

pub struct UnitMatmulFamily<TMM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TMM, RF)>,
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for UnitMatmulFamily<TMM, RF> {
    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        UnitMatmul<MP, TMM::Matmul<MP>, RF::Reader<MP::ES, TL>, RF::Reader<MP::ES, TR>>;

    type Input = StageInput;

    fn computation_resources(
        tiling_scheme: &TilingScheme,
    ) -> Result<ComputeResources, InvalidConfigError> {
        if let ComputeResources::Units(units) = TMM::computation_resources()? {
            Ok(ComputeResources::Units(
                units * tiling_scheme.stage_partitions_in_stage_mn(),
            ))
        } else {
            unreachable!("Unit matmul should not demand planes")
        }
    }

    fn setup(
        stage_input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        quantized: bool,
    ) -> Self::Config {
        let tile_input = TileSetupInput {
            vectorization: stage_input.stage_vectorization,
            tile_size: stage_input.tiling_scheme.tile_size,
        };
        let tile_config = TMM::setup(tile_input, problem, line_sizes, cube_dim);

        let compute_planes =
            <Self as StageMatmulFamily>::computation_resources(&stage_input.tiling_scheme)
                .unwrap_or_else(|e| panic!("{}", e))
                .as_plane_resources(tile_config.plane_dim())
                .unwrap_or_else(|e| panic!("{}", e))
                .get_count();
        let plane_role_config = PlaneRoleConfig::from_plane_roles(
            stage_input
                .load_specialization
                .to_plane_roles(compute_planes),
        );

        CommonStageConfig::new(
            tile_config,
            stage_input.tiling_scheme,
            quantized,
            stage_input.partition_buffering,
            stage_input.num_stages,
            plane_role_config,
        )
    }
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> MatmulChecker for UnitMatmulFamily<TMM, RF> {
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_units_needed = config.tiling_scheme().stage_partitions_in_stage_mn();
        let num_units = config.plane_dim() * config.num_main_flow_planes();

        if num_units != num_units_needed {
            return Err(Box::new(format!(
                "Error: Number of units {num_units} should be {num_units_needed}."
            )));
        }

        if config.partition_buffering() == PartitionBuffering::Double
            && config.tiling_scheme().tiles_in_stage_partition_n() < 2
        {
            return Err(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            ));
        }

        <TMM as MatmulChecker>::check_config(&config.tile_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tile_config)
    }
}
