// use crate::components::{
//     LoadingPrecomputeStrategy,
//     global::{
//         GlobalConfig, PlaneRoleConfig, SharedGlobalConfig, SpecializedLoadingSides,
//         read::ReaderMode,
//     },
//     stage::StageConfig,
// };

// #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// /// Configuration for the double buffering global matmul
// pub struct DoubleBufferingGlobalConfig<S: StageConfig> {
//     pub shared: SharedGlobalConfig<S>,
// }

// impl<S: StageConfig> GlobalConfig for DoubleBufferingGlobalConfig<S> {
//     // type StageConfig = S;
//     // type LhsReaderConfig = Self;
//     // type RhsReaderConfig = Self;

//     // fn lhs_reader_config(&self) -> Self::LhsReaderConfig {
//     //     *self
//     // }

//     // fn rhs_reader_config(&self) -> Self::RhsReaderConfig {
//     //     *self
//     // }

//     // fn stage_config(&self) -> Self::StageConfig {
//     //     self.stage_config
//     // }

//     // fn global_line_size(&self, ident: MatmulIdent) -> u32 {
//     //     self.stage_config.global_line_size(ident.into_stage())
//     // }

//     // fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
//     //     self.stage_config.matrix_layout(ident.into_stage())
//     // }

//     // fn plane_dim(&self) -> u32 {
//     //     self.stage_config.plane_dim()
//     // }

//     // fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
//     //     match ident {
//     //         MatmulIdent::Lhs => self.check_m_bounds,
//     //         MatmulIdent::Rhs => self.check_k_bounds,
//     //         MatmulIdent::Out => self.check_m_bounds,
//     //     }
//     // }

//     // fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
//     //     match ident {
//     //         MatmulIdent::Lhs => self.check_k_bounds,
//     //         MatmulIdent::Rhs => self.check_n_bounds,
//     //         MatmulIdent::Out => self.check_n_bounds,
//     //     }
//     // }

//     // fn check_k_bounds(&self) -> bool {
//     //     self.check_k_bounds
//     // }

//     // fn num_stages(&self, _ident: MatmulIdent) -> u32 {
//     //     2
//     // }

//     // fn cube_dim(&self) -> CubeDim {
//     //     CubeDim::new_2d(<Self as GlobalConfig>::plane_dim(self), self.num_planes)
//     // }

//     // fn role_rule_config(&self) -> RoleRuleConfig {
//     //     self.plane_role_config().rule
//     // }
// }

// // impl<S: StageConfig> GlobalReaderConfig for DoubleBufferingGlobalConfig<S> {
// // fn stage_memory_config(&self, ident: MatmulIdent) -> StageMemoryConfig {
// //     self.stage_config().stage_memory_config(ident.into_stage())
// // }

// // fn tiling_scheme(&self) -> TilingScheme {
// //     self.stage_config().tiling_scheme()
// // }

// // fn global_line_size(&self, ident: MatmulIdent) -> u32 {
// //     <Self as GlobalConfig>::global_line_size(self, ident)
// // }

// // fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
// //     <Self as GlobalConfig>::matrix_layout(self, ident)
// // }

// // fn num_loading_planes(&self, ident: MatmulIdent) -> u32 {
// //     self.specialized_loading_sides.num_loading_planes(
// //         self.plane_role_config().has_specialization(),
// //         ident,
// //         self.plane_role_config().plane_roles,
// //     )
// // }

// // fn plane_role_config(&self) -> PlaneRoleConfig {
// //     self.plane_role_config()
// // }

// // fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
// //     self.specialized_loading_sides
// // }

// // fn plane_dim(&self) -> u32 {
// //     <Self as GlobalConfig>::plane_dim(&self)
// // }

// // fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
// //     <Self as GlobalConfig>::check_row_bounds(self, ident)
// // }

// // fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
// //     <Self as GlobalConfig>::check_col_bounds(self, ident)
// // }

// // fn precompute_job(&self) -> bool {
// //     self.precompute_job.into()
// // }

// // fn num_stages(&self, ident: MatmulIdent) -> u32 {
// //     <Self as GlobalConfig>::num_stages(self, ident)
// // }

// // fn reader_mode(&self) -> ReaderMode {
// //     self.reader_mode
// // }

// // fn event_loading_mode(&self, _ident: MatmulIdent) -> EventLoadingMode {
// //     EventLoadingMode::Relaxed
// // }
// // }

// impl<S: StageConfig> DoubleBufferingGlobalConfig<S> {
//     pub fn from_shared_global_config(shared: SharedGlobalConfig<S>) -> Self {
//         Self { shared }
//     }

//     pub fn plane_role_config(&self) -> PlaneRoleConfig {
//         self.shared.stage_config.plane_role_config()
//     }
// }
