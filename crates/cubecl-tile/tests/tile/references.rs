//! CPU references for the tiled-arange tests. Each operand's physical buffer holds an arange in
//! tiled `[grid…, tile…]` order, so the value at a logical coordinate equals its physical flat
//! index. These recompute the expected matmul (or physical index) in that same order.
#![allow(dead_code)]

pub fn tiled_index(row: usize, col: usize, cols: usize, edge: usize) -> usize {
    let grid_c = cols / edge;
    let (gr, tr) = (row / edge, row % edge);
    let (gc, tc) = (col / edge, col % edge);
    ((gr * grid_c + gc) * edge + tr) * edge + tc
}

pub fn tiled_matmul(m: usize, n: usize, k: usize, edge: usize) -> Vec<f32> {
    let (grid_m, grid_n) = (m / edge, n / edge);
    let mut out = vec![0.0f32; m * n];
    for gm in 0..grid_m {
        for gn in 0..grid_n {
            for tm in 0..edge {
                for tn in 0..edge {
                    let (i, j) = (gm * edge + tm, gn * edge + tn);
                    let value = (0..k)
                        .map(|kk| {
                            tiled_index(i, kk, k, edge) as f32 * tiled_index(kk, j, n, edge) as f32
                        })
                        .sum::<f32>();
                    out[((gm * grid_n + gn) * edge + tm) * edge + tn] = value;
                }
            }
        }
    }
    out
}

pub fn batched_index(
    batch: usize,
    row: usize,
    col: usize,
    grid_row: usize,
    grid_col: usize,
    edge: usize,
    batch_edge: usize,
) -> usize {
    let (gb, tb) = (batch / batch_edge, batch % batch_edge);
    let (gr, tr) = (row / edge, row % edge);
    let (gc, tc) = (col / edge, col % edge);
    ((((gb * grid_row + gr) * grid_col + gc) * batch_edge + tb) * edge + tr) * edge + tc
}

pub fn batched_tiled_matmul(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    edge: usize,
    batch_edge: usize,
) -> Vec<f32> {
    let (grid_m, grid_n, grid_k) = (m / edge, n / edge, k / edge);
    let mut out = vec![0.0f32; b * m * n];
    for bb in 0..b {
        for gm in 0..grid_m {
            for gn in 0..grid_n {
                for tm in 0..edge {
                    for tn in 0..edge {
                        let (i, j) = (gm * edge + tm, gn * edge + tn);
                        let value = (0..k)
                            .map(|kk| {
                                batched_index(bb, i, kk, grid_m, grid_k, edge, batch_edge) as f32
                                    * batched_index(bb, kk, j, grid_k, grid_n, edge, batch_edge)
                                        as f32
                            })
                            .sum::<f32>();
                        out[batched_index(bb, i, j, grid_m, grid_n, edge, batch_edge)] = value;
                    }
                }
            }
        }
    }
    out
}

pub fn broadcast_matmul(b0: usize, b1: usize, t: usize) -> Vec<f32> {
    let lhs = |g0: usize, i: usize, kk: usize| (g0 * t * t + i * t + kk) as f32;
    let rhs = |g1: usize, kk: usize, j: usize| (g1 * t * t + kk * t + j) as f32;
    let mut out = vec![0.0f32; b0 * b1 * t * t];
    for u in 0..b0 * b1 {
        let (g0, g1) = (u / b1, u % b1);
        for i in 0..t {
            for j in 0..t {
                let value = (0..t)
                    .map(|kk| lhs(g0, i, kk) * rhs(g1, kk, j))
                    .sum::<f32>();
                out[u * t * t + i * t + j] = value;
            }
        }
    }
    out
}

pub fn nested_index(i: usize, j: usize, extent: usize, edges: &[usize]) -> usize {
    // Decompose a coordinate into its `[grid, level1, …, final]` digits.
    fn digits(c: usize, edges: &[usize]) -> Vec<usize> {
        let final_span: usize = edges.iter().product();
        let mut out = vec![c / final_span];
        let mut div = final_span;
        for &e in edges {
            div /= e;
            out.push((c / div) % e);
        }
        out
    }
    let grid = extent / edges.iter().product::<usize>();
    let (di, dj) = (digits(i, edges), digits(j, edges));
    let dims: Vec<usize> = std::iter::once(grid).chain(edges.iter().copied()).collect();
    let mut flat = 0;
    for (lvl, &dim) in dims.iter().enumerate() {
        flat = (flat * dim + di[lvl]) * dim + dj[lvl];
    }
    flat
}
