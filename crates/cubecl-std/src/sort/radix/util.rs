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
pub fn cube_ex_scan<N: Numeric>(
    input: N,
    #[comptime] cube_size: u32,
    #[comptime] min_wave_width: u32,
) -> N {
    let log_wave = u32::find_first_set(PLANE_DIM) - 1;

    let mut wave_scan =
        SharedMemory::<N>::new(comptime! {cube_size/min_wave_width});

    let mut local = plane_exclusive_sum(input);

    if UNIT_POS_PLANE == PLANE_DIM - 1 {
        wave_scan[UNIT_POS >> log_wave] = local + input;
    }

    sync_units();

    // comptime exp perhaps can eliminate branches
    if (min_wave_width >= comptime! {cube_size/min_wave_width})
        || (PLANE_DIM >= (cube_size >> log_wave))
    {
        // fast path
        if (UNIT_POS >> log_wave) == 0 {
            let cond = UNIT_POS < (cube_size >> log_wave);
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
        // for width < 16
        // (16 * 16 =256)
        let shared_use = cube_size >> log_wave;
        if plane_broadcast(UNIT_POS, 0) < shared_use {
            let cond = UNIT_POS < (cube_size >> log_wave);
            let prev = if cond {
                wave_scan[UNIT_POS]
            } else {
                wave_scan[0]
            };
            let temp = plane_inclusive_sum(prev);
            let temp = rotate::<N>(temp, -1, cond);
            if cond {
                wave_scan[UNIT_POS] = temp
            }
        }
        sync_units();

        let mut active = shared_use >> log_wave;
        let mut stride = PLANE_DIM;
        let mut stride_mask = stride - 1;

        while active > 1 {
            if plane_broadcast(UNIT_POS, 0) < active {
                let idx = UNIT_POS * stride;
                let cond = idx < active; //
                let prev =
                    if cond { wave_scan[idx] } else { wave_scan[0] };

                let temp = plane_inclusive_sum(prev);
                let temp = rotate::<N>(temp, -1, cond);
                if cond {
                    wave_scan[idx] = temp;
                }
            }
            sync_units();
            if plane_broadcast(UNIT_POS, 0) < active {
                let idx =
                    UNIT_POS & BitwiseNot::bitwise_not(stride_mask);
                let cond = UNIT_POS < shared_use;
                if (idx != 0) && (UNIT_POS != idx) && cond {
                    let temp = wave_scan[idx];
                    wave_scan[UNIT_POS] += temp;
                }
            }
            active >>= log_wave;
            stride <<= log_wave;
            stride_mask = stride - 1;
            sync_units();
        }
        if (UNIT_POS >> log_wave) != 0 {
            local += wave_scan[UNIT_POS >> log_wave];
        }
    }
    local
}

#[cube]
pub fn rotate<N: Numeric>(
    input: N,
    shift: i32,
    active: bool,
) -> N {
    let act = line_sum(Line::count_ones(plane_ballot(active)));
    plane_shuffle(
        input,
        (UNIT_POS_PLANE + (act as i32 + shift) as u32) % act,
    )
}
