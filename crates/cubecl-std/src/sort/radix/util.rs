use cubecl_core as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn memclean(
    input: &mut Array<u32>,
    #[comptime] keys: u32,
) {
    let mut idx = CUBE_POS * CUBE_DIM * keys + UNIT_POS;
    if CUBE_POS == CUBE_COUNT - 1 {
        for _ in 0..keys {
            if idx < input.len() {
                input[idx] = 0;
            }
            idx += CUBE_DIM;
        }
    } else {
        for _ in 0..keys {
            input[idx] = 0;
            idx += CUBE_DIM;
        }
    }
}

#[cube]
pub fn line_sum<N: Numeric>(l: Line<N>) -> N {
    let mut s = l[0];
    #[unroll]
    for i in 1..4 {
        s += l[i];
    }
    s
}

#[cube]
pub fn line_ffs(l: Line<u32>) -> u32 {
    let mut r = 0;
    for i in 0..4 {
        let pos = u32::find_first_set(l[i]);
        if pos != 0u32 {
            r += pos;
            break;
        }
        r += 32;
    }
    r
}

#[cube]
pub fn count_to_mask(count: u32) -> Line<u32> {
    let mut cnt = count;
    let mut res = Line::empty(4).fill(0);
    for i in 0..4 {
        if cnt >= 32 {
            res[i] = 0xFF_FF_FF_FFu32;
        } else {
            res[i] = (1 << cnt) - 1;
            break;
        }
        cnt -= 32;
    }
    res
}

#[cube]
pub fn cube_in_scan<N: Numeric>(
    input: N,
    #[comptime] cube_size: u32,
    #[comptime] wave_width: u32,
) -> N {
    // v3
    let log_wave = u32::find_first_set(PLANE_DIM) - 1;

    let mut wave_scan =
        SharedMemory::<N>::new(comptime! {cube_size/wave_width});

    let mut local = plane_inclusive_sum(input);

    if UNIT_POS_PLANE == PLANE_DIM - 1 {
        wave_scan[UNIT_POS >> log_wave] = local;
    }

    sync_units();

    if (wave_width >= comptime! {cube_size/wave_width})
        || (PLANE_DIM >= comptime! {cube_size/wave_width})
    {
        // fast path
        if UNIT_POS >> log_wave == 0 {
            let cond = UNIT_POS < (cube_size / wave_width);
            let prev = if cond {
                wave_scan[UNIT_POS]
            } else {
                wave_scan[0]
            };
            let temp = plane_exclusive_sum(prev);
            if cond {
                wave_scan[UNIT_POS] = temp
            }
        }
        sync_units();
        local += wave_scan[UNIT_POS >> log_wave];
    } else {
        // let active = cube_size >> log_wave;
        // slow path
        // let mut stride = 1;
        // let mut stride_mask = stride - 1;
        // while active != 0 {
        //     if plane_broadcast(UNIT_POS, 0) < active {
        //         let idx = UNIT_POS * stride + stride_mask;
        //         let cond = idx < (cube_size / wave_width);
        //         let prev = if cond { scan[idx] } else { scan[0] };
        //         let temp = plane_inclusive_sum(prev);
        //         if cond {
        //             should_add[idx] = temp - prev;
        //             scan[idx] = temp;
        //         }
        //     }
        //     sync_units();
        //     active >>= log_wave;
        //     stride <<= log_wave;
        //     stride_mask = stride - 1;

        //     local += should_add[(UNIT_POS | stride_mask) >> log_wave];
        // }
    }
    local
}

#[cube]
pub fn cube_ex_scan<N: Numeric>(
    input: N,
    #[comptime] cube_size: u32,
    #[comptime] wave_width: u32, // or min width
) -> N {
    let log_wave = u32::find_first_set(PLANE_DIM) - 1;

    let mut wave_scan =
        SharedMemory::<N>::new(comptime! {cube_size/wave_width});

    let mut local = plane_exclusive_sum(input);

    if UNIT_POS_PLANE == PLANE_DIM - 1 {
        wave_scan[UNIT_POS >> log_wave] = local + input;
    }

    sync_units();

    if (wave_width >= comptime! {cube_size/wave_width})
        || (PLANE_DIM >= comptime! {cube_size/wave_width})
    {
        // fast path
        if UNIT_POS >> log_wave == 0 {
            let cond = UNIT_POS < (cube_size / wave_width);
            let prev = if cond {
                wave_scan[UNIT_POS]
            } else {
                wave_scan[0]
            };
            let temp = plane_exclusive_sum(prev);
            if cond {
                wave_scan[UNIT_POS] = temp
            }
        }
        sync_units();
        local += wave_scan[UNIT_POS >> log_wave];
    } else {
        // todo
        // slow path
        // let active = cube_size >> log_wave;
        // let mut stride = 1;
        // let mut stride_mask = stride - 1;
        // while active != 0 {
        //     if plane_broadcast(UNIT_POS, 0) < active {
        //         let idx = UNIT_POS * stride + stride_mask;
        //         let cond = idx < (cube_size / wave_width);
        //         let prev = if cond { scan[idx] } else { scan[0] };
        //         let temp = plane_inclusive_sum(prev);
        //         if cond {
        //             should_add[idx] = temp - prev;
        //             scan[idx] = temp;
        //         }
        //     }
        //     sync_units();
        //     active >>= log_wave;
        //     stride <<= log_wave;
        //     stride_mask = stride - 1;

        //     local += should_add[(UNIT_POS | stride_mask) >> log_wave];
        // }
    }
    local
}
