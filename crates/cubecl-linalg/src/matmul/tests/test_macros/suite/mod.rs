//! Generates tests by combining parameters in the following order:
//!
//! 1. **Algorithm**
//!    - `kind`: high-level matmul category (e.g., `accelerated`, `tma`, `unit`, `quantized`)
//!    - `algorithm`: compute/loading strategy (e.g., `double_buffering_tilewise`, `simply_cyclic`)
//! 2. **Data**
//!    - `precision`: data type (e.g., `f16`, `f32`)
//!    - `layouts`: operand layouts (row-major (r) or column-major (c) for lhs and rhs)
//! 3. **Execution Shape**
//!    - `tile`: instruction tile dimensions in M/N/K
//!    - `stage`: shared memory shape (number of tiles in M/N/K)
//! 4. **Problem**
//!    - `problem`: actual matrix dimensions M/N/K

mod algorithm;
mod kind;
pub mod launch;
mod layout;
mod precision;
mod problem;
mod stage;
mod tile;
