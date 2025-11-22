use cubecl_core::{self as cubecl, prelude::*};
use cubecl_std::tensor::TensorHandle;
use cubecl::frontend::TensorHandleRef;
use std::collections::HashSet;
use std::env;
use std::sync::{LazyLock, Mutex};

// ================================
// Constants & tuning parameters
// ================================

/// Tile size optimized for 4-element vectorized loads (mov4)
const TILE_SIZE_MOV4: u32 = 32;
/// Tile size optimized for 2-element vectorized loads (mov2)
const TILE_SIZE_MOV2: u32 = 64;
/// Number of threads per tile column for cooperative loading
const BLOCK_ROWS: u32 = 8;

// ===========================================================
// Host-side utility functions for shape and stride calculations
// ===========================================================

/// Compute output shape after applying permutation `axes` to `input_shape`.
///
/// Example: `infer_output_shape(&[2,3,4], &[1,0,2])` returns `[3,2,4]`
fn infer_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    assert_eq!(
        axes.len(),
        input_shape.len(),
        "axes length must match input shape"
    );
    axes.iter().map(|&a| input_shape[a]).collect()
}

/// Extract (batch, height, width) dimensions for batch transpose kernels.
///
/// Converts 2D or 3D input shapes into a standardized 3D format for transpose kernels.
#[allow(dead_code)]
fn infer_batch_transpose_shape(input_shape: &[usize], _axes: &[usize]) -> (u32, u32, u32) {
    match input_shape.len() {
        2 => {
            // [H, W] → treat as single batch
            (1, input_shape[0] as u32, input_shape[1] as u32)
        }
        3 => {
            // [B, H, W] → batched 2D
            (
                input_shape[0] as u32,
                input_shape[1] as u32,
                input_shape[2] as u32,
            )
        }
        _ => panic!(
            "infer_batch_transpose_shape only supports rank 2 or 3, got rank {}",
            input_shape.len()
        ),
    }
}

/// Result of dimension folding optimization
#[derive(Debug, Clone)]
struct FoldedPermutation {
    /// Folded shape (lower rank, merged contiguous dims)
    folded_shape: Vec<usize>,
    /// Permutation in terms of folded dimensions
    folded_axes: Vec<usize>,
}

/// Fold contiguous dimensions to simplify permutation.
///
/// This is a CRITICAL optimization that can turn complex high-rank permutations
/// into simple 2D transposes.
///
/// Algorithm:
/// 1. Identify runs of dimensions that are contiguous in memory (stride[i] == stride[i+1] * shape[i+1])
/// 2. Merge those dimensions by multiplying their sizes
/// 3. Update the axes permutation to work on the folded dimensions
///
/// Example:
/// - Input: shape=[8, 16, 32, 64], strides=[32768, 2048, 64, 1], axes=[0, 3, 2, 1]
/// - Last two dims are contiguous: stride[2]=64 == stride[3]*shape[3] = 1*64
/// - Fold into: shape=[8, 16, 2048], strides=[32768, 2048, 1], axes=[0, 2, 1]
/// - Now it's a simple 3D batch transpose!
fn fold_contiguous_dimensions(
    input_shape: &[usize],
    input_strides: &[usize],
    axes: &[usize],
) -> FoldedPermutation {
    let rank = input_shape.len();

    if rank <= 1 {
        return FoldedPermutation {
            folded_shape: input_shape.to_vec(),
            folded_axes: axes.to_vec(),
        };
    }

    // Find contiguous runs in the INPUT tensor
    // A run is contiguous if stride[i] == stride[i+1] * shape[i+1]
    let mut is_contiguous_with_next = vec![false; rank];
    for i in 0..rank - 1 {
        is_contiguous_with_next[i] = input_strides[i] == input_strides[i + 1] * input_shape[i + 1];
    }

    // Build folded dimensions by merging contiguous runs
    let mut folded_shape = Vec::new();
    let mut old_to_new_axis = vec![0usize; rank]; // Maps old axis index to folded axis index

    let mut i = 0;
    while i < rank {
        let start = i;

        // Extend run while contiguous
        while i < rank - 1 && is_contiguous_with_next[i] {
            i += 1;
        }

        // Merge dimensions [start..=i]
        let merged_size: usize = (start..=i).map(|j| input_shape[j]).product();
        folded_shape.push(merged_size);

        // All axes in this run map to the same folded axis
        let folded_idx = folded_shape.len() - 1;
        for item in old_to_new_axis.iter_mut().take(i + 1).skip(start) {
            *item = folded_idx;
        }

        i += 1;
    }

    // Now we need to check if the PERMUTATION preserves contiguous runs
    // If axes permutes within a folded group, we can't use the folding
    // Example: if dims 2,3 were folded but axes=[0,1,3,2], we can't fold
    // Also: if dims are folded but get REORDERED, we can't fold (e.g., axes=[1,0] for 2D)

    // Check if axes respects folded groups
    let mut axes_respects_folding = true;
    for fold_idx in 0..folded_shape.len() {
        // Find all old axes that map to this folded axis
        let old_axes_in_group: Vec<usize> = (0..rank)
            .filter(|&i| old_to_new_axis[i] == fold_idx)
            .collect();

        if old_axes_in_group.len() > 1 {
            // Check if these axes appear in the SAME ORDER in the permutation
            // Find their positions in the axes array
            let mut positions: Vec<usize> = old_axes_in_group
                .iter()
                .map(|&old_ax| axes.iter().position(|&a| a == old_ax).unwrap())
                .collect();

            // They must be consecutive and in ascending order
            // This ensures the folded group stays together and in order
            positions.sort_unstable();
            for j in 0..positions.len() - 1 {
                if positions[j] + 1 != positions[j + 1] {
                    axes_respects_folding = false;
                    break;
                }
            }

            // Verify axes are in ascending order at those positions.
            // Example: for axes=[1,0], positions=[0,1] but old_axes_in_group=[0,1],
            // we need axes[positions[0]] < axes[positions[1]]
            if axes_respects_folding {
                for j in 0..old_axes_in_group.len() - 1 {
                    let pos_j = axes
                        .iter()
                        .position(|&a| a == old_axes_in_group[j])
                        .unwrap();
                    let pos_jp1 = axes
                        .iter()
                        .position(|&a| a == old_axes_in_group[j + 1])
                        .unwrap();
                    if pos_j > pos_jp1 {
                        // Axes are reversed or out of order within the group - folding not possible
                        axes_respects_folding = false;
                        break;
                    }
                }
            }
        }
    }

    if !axes_respects_folding {
        // Folding would produce incorrect results - return original dimensions
        return FoldedPermutation {
            folded_shape: input_shape.to_vec(),
            folded_axes: axes.to_vec(),
        };
    }

    // Build folded axes: for each position in axes, find which folded group it belongs to
    // and use the first axis from that group
    let mut folded_axes = Vec::new();
    let mut seen_folded = vec![false; folded_shape.len()];

    for &ax in axes {
        let folded_idx = old_to_new_axis[ax];
        if !seen_folded[folded_idx] {
            folded_axes.push(folded_idx);
            seen_folded[folded_idx] = true;
        }
    }

    FoldedPermutation {
        folded_shape,
        folded_axes,
    }
}
// This is the beginning of the permute.rs code
// All the kernels are here
// ...
// ... all the way to the end
// ...
fn main() {
    println!("This example is empty. The permute code is in lib.rs");
}