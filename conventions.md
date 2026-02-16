# CubeCL Naming Conventions

This document describes the naming conventions used throughout the CubeCL codebase.

## Tensor Dimensions

- Use **`axis`** (not `dim` or `dimension`) when referring to a specific dimension of a tensor.
  - `tensor.stride(axis)`, `tensor.shape(axis)`, `tensor.coordinate(index, axis)`

## Counts

- Use **`_count` suffix** (not `num_` prefix) for quantities.
  - `streaming_multiprocessor_count`, `cpu_core_count`, `tensor_core_count`
  - `elem_count`, `cube_count`, `meta_count`
  - Constants: `SM_COUNT_APPROX`

## Line Size

- Use **`line_size`** (not `vectorization` or `vectorization_factor`) for the number of
  elements packed into a line.
  - `tensor.line_size()`, `find_line_size()`, `tensor_line_size_parallel()`
  - `tensor_vectorization_factor()` remains only as a deprecated compatibility alias.

## Tensor Ordering

- **`RowMajor`** / **`ColMajor`** are the primary names for matrix layouts.
- **`DecreasingOrder`** / **`IncreasingOrder`** are available as aliases:
  - `MatrixLayout::IncreasingOrder` = `MatrixLayout::ColMajor`
  - `MatrixLayout::DecreasingOrder` = `MatrixLayout::RowMajor`

## Coordinates and Offsets

- Use **`offset`** for linear buffer/slice positions.
- Use **`coordinate`** for multi-dimensional tensor positions.

## Topology Constants

- Use **`POS`** suffix for positions: `UNIT_POS`, `CUBE_POS`, `PLANE_POS`.
- Use **`DIM`** suffix for topology dimensions: `CUBE_DIM`, `PLANE_DIM`.

## Type Naming

- Use **postfix suffixes** for type categories:
  - `*Error` for error types: `LineSizeError`, `LaunchError`
  - `*Strategy` for strategy types: `ReadingStrategy`
  - `*Expand` for expand/meta types: `TensorExpand`, `SliceExpand`
