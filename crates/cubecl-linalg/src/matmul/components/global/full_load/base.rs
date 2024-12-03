use crate::matmul::components::global::unloader::Unloader;
use crate::matmul::components::global::{Config as _, Loader, LoadingStrategy};
use crate::matmul::components::stage;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{config::MatmulConfig, global::ZeroAccumulatorLoader};
use crate::matmul::components::{global, MatmulProblem};
use crate::matmul::components::{Ident, MatrixLayout};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::{LhsLoader, RhsLoader};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _acc: PhantomData<EA>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

// #[cube(debug)]
// impl<EG, ES, EA, SMM, LL, RL> global::Matmul<EG, ES> for Matmul<EG, ES, EA, SMM, LL, RL>
// where
//     EG: Numeric,
//     ES: Numeric,
//     EA: Numeric,
//     SMM: stage::Matmul<ES, EG, EA, LhsReader = LhsReader<ES>, RhsReader = RhsReader<ES>>,
//     LL: LoadingStrategy<EG, ES>,
//     RL: LoadingStrategy<EG, ES>,
// {
//     type LhsLoader = LhsLoader<EG, ES, LL>;
//     type RhsLoader = RhsLoader<EG, ES, RL>;
//     type AccumulatorLoader = ZeroAccumulatorLoader;
//     type Out = Unloader<EG>;
//     type Accumulator = SMM::Accumulator;

//     fn execute(
//         mut lhs_loader: Self::LhsLoader,
//         mut rhs_loader: Self::RhsLoader,
//         mut out_unloader: Self::Out,
//         acc: &mut Self::Accumulator,
//         k_range: (u32, u32),
//         #[comptime] config: Self::Config,
//     ) {
//         let k_step = SMM::K;
//         let range = k_range.1 - k_range.0;
//         let num_stages = (range + k_step - 1) / k_step;

//         let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
//         SMM::zero_accumulator(acc, config.to_smm_config());

//         let mut lhs_curr = Self::LhsLoader::init_buffer::<Self::Config>(config);
//         let mut rhs_curr = Self::RhsLoader::init_buffer::<Self::Config>(config);
//         let mut lhs_next = Self::LhsLoader::init_buffer::<Self::Config>(config);
//         let mut rhs_next = Self::RhsLoader::init_buffer::<Self::Config>(config);

//         Self::LhsLoader::fetch_global::<Self::Config>(&mut lhs_loader, &mut lhs_curr, config);
//         Self::RhsLoader::fetch_global::<Self::Config>(&mut rhs_loader, &mut rhs_curr, config);

//         for _ in 0..num_stages {
//             sync_units();

//             Self::LhsLoader::to_next_stage::<Self::Config>(&mut lhs_loader, config);
//             Self::RhsLoader::to_next_stage::<Self::Config>(&mut rhs_loader, config);

//             Self::LhsLoader::fetch_global::<Self::Config>(&mut lhs_loader, &mut lhs_next, config);
//             Self::RhsLoader::fetch_global::<Self::Config>(&mut rhs_loader, &mut rhs_next, config);

//             let lhs_stage_reader =
//                 &LhsLoader::fill_stage::<Self::Config>(&mut lhs_loader, &lhs_curr, config);
//             let rhs_stage_reader =
//                 &RhsLoader::fill_stage::<Self::Config>(&mut rhs_loader, &rhs_curr, config);

//             sync_units();

//             lhs_curr.swap(&mut lhs_next);
//             rhs_curr.swap(&mut rhs_next);

//             SMM::execute(
//                 lhs_stage_reader,
//                 rhs_stage_reader,
//                 &mut lhs_tile,
//                 &mut rhs_tile,
//                 acc,
//                 config.to_smm_config(),
//             );
//         }

//         sync_units();

//         SMM::read_accumulator::<Self::Out, Self::Config>(
//             acc,
//             &mut out_unloader,
//             config.to_smm_config(),
//             config,
//         );
//     }

//     fn init_lhs_loader(
//         lhs: &Tensor<Line<EG>>,
//         x_offset: u32,
//         y_offset: u32,
//         batch_offset: u32,
//         #[comptime] config: Self::Config,
//     ) -> Self::LhsLoader {
//         LhsLoader::new::<<Self::Config as global::Config>::SmmConfig>(
//             lhs,
//             x_offset,
//             y_offset,
//             batch_offset,
//             config,
//         )
//     }

//     fn init_rhs_loader(
//         rhs: &Tensor<Line<EG>>,
//         x_offset: u32,
//         y_offset: u32,
//         batch_offset: u32,
//         #[comptime] config: Self::Config,
//     ) -> Self::RhsLoader {
//         RhsLoader::new::<<Self::Config as global::Config>::SmmConfig>(
//             rhs,
//             x_offset,
//             y_offset,
//             batch_offset,
//             config,
//         )
//     }

//     fn init_unloader(
//         out: &mut Tensor<Line<EG>>,
//         x_offset: u32,
//         y_offset: u32,
//         batch_offset: u32,
//     ) -> Self::Out {
//         Self::Out::new(out, x_offset, y_offset, batch_offset)
//     }

//     fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
//         SMM::init_accumulator(config.to_smm_config())
//     }

//     fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
//         SMM::zero_accumulator(acc, config.to_smm_config());
//     }
// }

impl<EG, ES, EA, SMM, LL, RL> global::Matmul<EG, ES> for Matmul<EG, ES, EA, SMM, LL, RL>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA, LhsReader = LhsReader<ES>, RhsReader = RhsReader<ES>>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
{
    type LhsLoader = LhsLoader<EG, ES, LL>;
    type RhsLoader = RhsLoader<EG, ES, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<EG>;
    type Accumulator = SMM::Accumulator;
    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        config: Self::Config,
    ) {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());
        let mut lhs_curr = Self::LhsLoader::init_buffer::<Self::Config>(config);
        let mut rhs_curr = Self::RhsLoader::init_buffer::<Self::Config>(config);
        let mut lhs_next = Self::LhsLoader::init_buffer::<Self::Config>(config);
        let mut rhs_next = Self::RhsLoader::init_buffer::<Self::Config>(config);
        Self::LhsLoader::fetch_global::<Self::Config>(&mut lhs_loader, &mut lhs_curr, config);
        Self::RhsLoader::fetch_global::<Self::Config>(&mut rhs_loader, &mut rhs_curr, config);
        for _ in 0..num_stages {
            sync_units();
            Self::LhsLoader::to_next_stage::<Self::Config>(&mut lhs_loader, config);
            Self::RhsLoader::to_next_stage::<Self::Config>(&mut rhs_loader, config);
            Self::LhsLoader::fetch_global::<Self::Config>(&mut lhs_loader, &mut lhs_next, config);
            Self::RhsLoader::fetch_global::<Self::Config>(&mut rhs_loader, &mut rhs_next, config);
            let lhs_stage_reader =
                &LhsLoader::fill_stage::<Self::Config>(&mut lhs_loader, &lhs_curr, config);
            let rhs_stage_reader =
                &RhsLoader::fill_stage::<Self::Config>(&mut rhs_loader, &rhs_curr, config);
            sync_units();
            lhs_curr.swap(&mut lhs_next);
            rhs_curr.swap(&mut rhs_next);
            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );
        }
        sync_units();
        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }
    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        config: Self::Config,
    ) -> Self::LhsLoader {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        LhsLoader::new::<<Self::Config as global::Config>::SmmConfig>(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            config,
        )
    }
    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        config: Self::Config,
    ) -> Self::RhsLoader {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        RhsLoader::new::<<Self::Config as global::Config>::SmmConfig>(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            config,
        )
    }
    fn init_unloader(
        out: &mut Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Out {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        Self::Out::new(out, x_offset, y_offset, batch_offset)
    }
    fn init_accumulator(config: Self::Config) -> Self::Accumulator {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        SMM::init_accumulator(config.to_smm_config())
    }
    fn zero_accumulator(acc: &mut Self::Accumulator, config: Self::Config) {
        use cubecl::prelude::{CubeIndex as _, CubeIndexMut as _};
        SMM::zero_accumulator(acc, config.to_smm_config());
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_execute(
        context: &mut cubecl::prelude::CubeContext,
        lhs_loader: <Self::LhsLoader as cubecl::prelude::CubeType>::ExpandType,
        rhs_loader: <Self::RhsLoader as cubecl::prelude::CubeType>::ExpandType,
        out_unloader: <Self::Out as cubecl::prelude::CubeType>::ExpandType,
        acc: <Self::Accumulator as cubecl::prelude::CubeType>::ExpandType,
        k_range: <(u32, u32) as cubecl::prelude::CubeType>::ExpandType,
        config: Self::Config,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            let k_step = SMM::K;
            let range = {
                let _lhs = k_range.clone().1.clone();
                let _rhs = k_range.0.clone();
                cubecl::frontend::sub::expand(context, _lhs, _rhs)
            };
            let num_stages = {
                let _lhs = {
                    let _lhs = {
                        let _lhs = range;
                        let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(k_step);
                        cubecl::frontend::add::expand(context, _lhs, _rhs)
                    };
                    let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(1);
                    cubecl::frontend::sub::expand(context, _lhs, _rhs)
                };
                let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(k_step);
                cubecl::frontend::div::expand(context, _lhs, _rhs)
            };
            let __tuple_destructure_init = {
                let _arg_0 = config.to_smm_config();
                SMM::__expand_init_tile_inputs(context, _arg_0.into())
            };
            let mut lhs_tile = {
                let _init = __tuple_destructure_init.clone().0.clone();
                cubecl::frontend::Init::init(_init, context)
            };
            let mut rhs_tile = {
                let _init = __tuple_destructure_init.1.clone();
                cubecl::frontend::Init::init(_init, context)
            };
            {
                let _arg_0 = acc.clone();
                let _arg_1 = config.to_smm_config();
                SMM::__expand_zero_accumulator(context, _arg_0.into(), _arg_1.into())
            };
            let mut lhs_curr = {
                let _init = {
                    let _arg_0 = config.clone();
                    Self::LhsLoader::__expand_init_buffer::<Self::Config>(context, _arg_0.into())
                };
                cubecl::frontend::Init::init(_init, context)
            };
            let mut rhs_curr = {
                let _init = {
                    let _arg_0 = config.clone();
                    Self::RhsLoader::__expand_init_buffer::<Self::Config>(context, _arg_0.into())
                };
                cubecl::frontend::Init::init(_init, context)
            };
            let mut lhs_next = {
                let _init = {
                    let _arg_0 = config.clone();
                    Self::LhsLoader::__expand_init_buffer::<Self::Config>(context, _arg_0.into())
                };
                cubecl::frontend::Init::init(_init, context)
            };
            let mut rhs_next = {
                let _init = {
                    let _arg_0 = config.clone();
                    Self::RhsLoader::__expand_init_buffer::<Self::Config>(context, _arg_0.into())
                };
                cubecl::frontend::Init::init(_init, context)
            };
            {
                let _arg_0 = lhs_loader.clone();
                let _arg_1 = lhs_curr.clone();
                let _arg_2 = config.clone();
                Self::LhsLoader::__expand_fetch_global::<Self::Config>(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                )
            };
            {
                let _arg_0 = rhs_loader.clone();
                let _arg_1 = rhs_curr.clone();
                let _arg_2 = config.clone();
                Self::RhsLoader::__expand_fetch_global::<Self::Config>(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                )
            };
            {
                let _range = {
                    let _start = 0;
                    let _end = num_stages;
                    cubecl::frontend::RangeExpand::new(_start.into(), _end.into(), false)
                };
                let _unroll = false;
                cubecl::frontend::branch::for_expand(context, _range, _unroll, |context, _| {
                    {
                        sync_units::expand(context)
                    };
                    {
                        let _arg_0 = lhs_loader.clone();
                        let _arg_1 = config.clone();
                        Self::LhsLoader::__expand_to_next_stage::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                        )
                    };
                    {
                        let _arg_0 = rhs_loader.clone();
                        let _arg_1 = config.clone();
                        Self::RhsLoader::__expand_to_next_stage::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                        )
                    };
                    {
                        let _arg_0 = lhs_loader.clone();
                        let _arg_1 = lhs_next.clone();
                        let _arg_2 = config.clone();
                        Self::LhsLoader::__expand_fetch_global::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                        )
                    };
                    {
                        let _arg_0 = rhs_loader.clone();
                        let _arg_1 = rhs_next.clone();
                        let _arg_2 = config.clone();
                        Self::RhsLoader::__expand_fetch_global::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                        )
                    };
                    let lhs_stage_reader = {
                        let _arg_0 = lhs_loader.clone();
                        let _arg_1 = lhs_curr.clone();
                        let _arg_2 = config.clone();
                        LhsLoader::__expand_fill_stage::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                        )
                    };
                    let rhs_stage_reader = {
                        let _arg_0 = rhs_loader.clone();
                        let _arg_1 = rhs_curr.clone();
                        let _arg_2 = config.clone();
                        RhsLoader::__expand_fill_stage::<Self::Config>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                        )
                    };
                    {
                        sync_units::expand(context)
                    };
                    {
                        let _arg_0 = lhs_next.clone();
                        lhs_curr
                            .clone()
                            .__expand_swap_method(context, _arg_0.into())
                    };
                    {
                        let _arg_0 = rhs_next.clone();
                        rhs_curr
                            .clone()
                            .__expand_swap_method(context, _arg_0.into())
                    };
                    {
                        let _arg_0 = lhs_stage_reader;
                        let _arg_1 = rhs_stage_reader;
                        let _arg_2 = lhs_tile.clone();
                        let _arg_3 = rhs_tile.clone();
                        let _arg_4 = acc.clone();
                        let _arg_5 = config.to_smm_config();
                        SMM::__expand_execute(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                            _arg_3.into(),
                            _arg_4.into(),
                            _arg_5.into(),
                        )
                    };
                    ()
                });
            };
            {
                sync_units::expand(context)
            };
            {
                let _arg_0 = acc;
                let _arg_1 = out_unloader;
                let _arg_2 = config.to_smm_config();
                let _arg_3 = config.clone();
                SMM::__expand_read_accumulator::<Self::Out, Self::Config>(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                    _arg_3.into(),
                )
            };
            ()
        }
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_init_lhs_loader(
        context: &mut cubecl::prelude::CubeContext,
        lhs: <Tensor<Line<EG>> as cubecl::prelude::CubeType>::ExpandType,
        x_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        y_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        batch_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        config: Self::Config,
    ) -> <Self::LhsLoader as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _arg_0 = lhs;
                let _arg_1 = x_offset;
                let _arg_2 = y_offset;
                let _arg_3 = batch_offset;
                let _arg_4 = config.clone();
                LhsLoader::__expand_new::<<Self::Config as global::Config>::SmmConfig>(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                    _arg_3.into(),
                    _arg_4.into(),
                )
            }
        }
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_init_rhs_loader(
        context: &mut cubecl::prelude::CubeContext,
        rhs: <Tensor<Line<EG>> as cubecl::prelude::CubeType>::ExpandType,
        x_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        y_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        batch_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        config: Self::Config,
    ) -> <Self::RhsLoader as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _arg_0 = rhs;
                let _arg_1 = x_offset;
                let _arg_2 = y_offset;
                let _arg_3 = batch_offset;
                let _arg_4 = config.clone();
                RhsLoader::__expand_new::<<Self::Config as global::Config>::SmmConfig>(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                    _arg_3.into(),
                    _arg_4.into(),
                )
            }
        }
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_init_unloader(
        context: &mut cubecl::prelude::CubeContext,
        out: <Tensor<Line<EG>> as cubecl::prelude::CubeType>::ExpandType,
        x_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        y_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        batch_offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
    ) -> <Self::Out as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _arg_0 = out;
                let _arg_1 = x_offset;
                let _arg_2 = y_offset;
                let _arg_3 = batch_offset;
                Self::Out::__expand_new(
                    context,
                    _arg_0.into(),
                    _arg_1.into(),
                    _arg_2.into(),
                    _arg_3.into(),
                )
            }
        }
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_init_accumulator(
        context: &mut cubecl::prelude::CubeContext,
        config: Self::Config,
    ) -> <Self::Accumulator as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _arg_0 = config.to_smm_config();
                SMM::__expand_init_accumulator(context, _arg_0.into())
            }
        }
    }
    #[allow(unused, clone_on_copy, clippy::all)]
    fn __expand_zero_accumulator(
        context: &mut cubecl::prelude::CubeContext,
        acc: <Self::Accumulator as cubecl::prelude::CubeType>::ExpandType,
        config: Self::Config,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _arg_0 = acc;
                let _arg_1 = config.to_smm_config();
                SMM::__expand_zero_accumulator(context, _arg_0.into(), _arg_1.into())
            };
            ()
        }
    }
}

impl<EG, ES, EA, SMM, LL, RL> MatmulKernel<EG, EG> for Matmul<EG, ES, EA, SMM, LL, RL>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
{
    type Config = Config<SMM::Config>;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let smm_config = SMM::make_config(problem, cube_dim, cube_count, advanced_config);

        Config::new(
            smm_config,
            problem.m as u32 % SMM::M != 0,
            problem.n as u32 % SMM::N != 0,
            problem.k as u32 % SMM::K != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the full load matmul
pub struct Config<S: stage::Config> {
    smm_config: S,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl<S: stage::Config> global::Config for Config<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn global_line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn stage_line_size(&self, ident: Ident) -> u32 {
        self.smm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim> {
        self.smm_config.stage_dim(ident)
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => self.smm_config.layout(Ident::Out),
        }
    }

    fn num_planes(&self) -> u32 {
        self.smm_config.num_planes()
    }

    fn plane_dim(&self) -> u32 {
        self.smm_config.plane_dim()
    }

    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
        self.smm_config.tiling_order(ident)
    }

    fn check_m_bounds(&self) -> bool {
        self.check_m_bounds
    }

    fn check_n_bounds(&self) -> bool {
        self.check_n_bounds
    }

    fn check_k_bounds(&self) -> bool {
        self.check_k_bounds
    }

    fn transpose_load(&self, ident: Ident) -> bool {
        self.layout(ident) != self.smm_config.layout(ident)
    }
}

impl<S: stage::Config> MatmulConfig for Config<S> {}

impl<S: stage::Config> Config<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smm_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            smm_config,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
