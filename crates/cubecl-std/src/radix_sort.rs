use std::mem;

use cubecl_core as cubecl;

use cubecl::{frontend::Atomic, prelude::*, server::Handle};

const FLAG_MASK: u32 = 0x_C0_00_00_00u32; // 11
const VALUE_MASK: u32 = 0x_3F_FF_FF_FFu32; // 00

const FLAG_REDUCTION: u32 = 0x_40_00_00_00u32; //Flag value indicating reduction of a partition tile is ready
                                               // 10
const FLAG_INCLUSIVE: u32 = 0x_80_00_00_00u32;

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
pub trait AsRadix:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + Sized
    + CubePrimitive
{
    type Radix: RadixData;
    fn as_radix(this: &Self) -> Self::Radix;
}

#[cube]
impl AsRadix for u32 {
    type Radix = u32;
    fn as_radix(this: &Self) -> Self::Radix {
        *this
    }
}

#[cube]
pub trait RadixData:
    CubeType<ExpandType = ExpandElementTyped<Self>> + Sized
{
    const BYTECNT: u32;
    fn shift_mask(
        this: &Self,
        shift: u32,
        mask: u32,
    ) -> u32;
}

#[cube]
impl<T: Int> RadixData for T {
    const BYTECNT: u32 = T::BITS / 8;
    fn shift_mask(
        this: &Self,
        shift: u32,
        mask: u32,
    ) -> u32 {
        u32::cast_from(*this >> T::cast_from(shift)) & mask
    }
}

#[cube(launch)]
fn global_histogram<N: AsRadix>(
    input: &Array<N>,
    global_histogram: &Array<Atomic<u32>>,
    block_cnt: u32,
    #[comptime] key_per_thread: u32,
) where
    Array<N>: LaunchArg,
{
    let len = input.len();
    let shared = SharedMemory::<Atomic<u32>>::new(
        comptime! {N::Radix::BYTECNT<<8},
    );
    // clean mem
    #[unroll]
    for i in 0..N::Radix::BYTECNT {
        Atomic::store(&shared[UNIT_POS + i * 256], 0);
    }
    sync_units();
    let mut ele_pos = CUBE_POS * CUBE_DIM * key_per_thread + UNIT_POS;
    #[unroll]
    for _ in 0..key_per_thread {
        if ele_pos < len {
            let item = N::as_radix(&input[ele_pos]);
            #[unroll]
            for offset in 0..N::Radix::BYTECNT {
                Atomic::add(
                    &shared[N::Radix::shift_mask(
                        &item,
                        8 * offset,
                        0xFFu32,
                    ) + offset * 256u32],
                    1,
                );
            }
        }
        ele_pos += CUBE_DIM;
    }
    sync_units();
    #[unroll]
    for i in 0..N::Radix::BYTECNT {
        Atomic::add(
            &global_histogram
                [UNIT_POS + (((block_cnt + 1) * i) << 8)],
            // &global_histogram[UNIT_POS + (block_cnt + 1 * i) << 8],
            Atomic::load(&shared[UNIT_POS + (i << 8)]),
        );
    }
}

#[cube(launch)]
fn ex_scan(
    input: &mut Array<u32>,
    block_cnt: u32,
    #[comptime] cube_size: u32, // radix
    #[comptime] wave_width: u32,
) {
    let pos = ((block_cnt + 1) * CUBE_POS << 8) + UNIT_POS;
    // let pos = (block_cnt + 1) * CUBE_POS << 8 + UNIT_POS;
    input[pos] =
        ex_block_reduce::<u32>(input[pos], cube_size, wave_width)
            | FLAG_INCLUSIVE;
}

#[cube(launch)]
fn one_sweep<N: RadixSort + CubePrimitive>(
    input: &Array<N>,
    output: &mut Array<N>,
    global_histogram: &Array<Atomic<u32>>,
    index: &Array<Atomic<u32>>,
    radix_shift: u32,
    #[comptime] key_per_thread: u32,
    #[comptime] min_wave_width: u32,
) {
    let pass = radix_shift / 8;

    N::one_sweep(
        min_wave_width,
        input,
        output,
        global_histogram,
        &index[pass],
        key_per_thread,
        radix_shift,
    );
}

pub async fn radix_sort<R: Runtime, N: RadixSort + AsRadix>(
    client: &cubecl::prelude::ComputeClient<R::Server, R::Channel>,
    len: usize,
    mut input: Handle,
    mut buffer: Handle,
) -> Handle {
    unsafe {
        let width = 32;
        let block_size = 256;
        let radix = 256;

        let bytecnt = N::Radix::BYTECNT as usize;

        let key_per_thread = 15;
        let key_per_block = (block_size * key_per_thread) as usize;

        let block_cnt =
            ((len + key_per_block - 1) / key_per_block) as u32;

        let global_histogram_handle = {
            let len = radix
                * (block_cnt as usize + 1)
                * mem::size_of::<u32>()
                * bytecnt;
            let handle = client.empty(len);
            let keys = 64;
            memclean::launch::<R>(
                client,
                CubeCount::Static(
                    (len as u32 + (1024 * keys) - 1) / (1024 * keys),
                    1,
                    1,
                ),
                CubeDim::new(1024, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&handle, len, 1),
                keys,
            );
            handle
        };

        let global_histogram = ArrayArg::from_raw_parts::<Atomic<u32>>(
            &global_histogram_handle,
            radix * bytecnt,
            1,
        );
        global_histogram::launch::<N, R>(
            client,
            CubeCount::Static(block_cnt, 1, 1),
            CubeDim::new(block_size, 1, 1),
            ArrayArg::from_raw_parts::<N>(&input, len, 1),
            global_histogram,
            ScalarArg::new(block_cnt),
            key_per_thread,
        );

        ex_scan::launch::<R>(
            client,
            CubeCount::Static(bytecnt as u32, 1, 1),
            CubeDim::new(block_size, 1, 1),
            ArrayArg::from_raw_parts::<u32>(
                &global_histogram_handle,
                radix * bytecnt,
                1,
            ),
            ScalarArg::new(block_cnt),
            block_size,
            width,
        );

        let index_handle =
            client.create(u32::as_bytes(&vec![0u32; bytecnt]));

        for i in 0..bytecnt {
            let offset = (block_cnt + 1) as usize
                * radix
                * i
                * size_of::<u32>();
            let global_histogram_handle = global_histogram_handle
                .clone()
                .offset_start(offset as u64);

            one_sweep::launch::<N, R>(
                client,
                CubeCount::Static(block_cnt, 1, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<N>(&input, len, 1),
                ArrayArg::from_raw_parts::<N>(&buffer, len, 1),
                ArrayArg::from_raw_parts::<Atomic<u32>>(
                    &global_histogram_handle,
                    radix * bytecnt,
                    1,
                ),
                ArrayArg::from_raw_parts::<N>(
                    &index_handle,
                    bytecnt,
                    1,
                ),
                ScalarArg::new(i as u32 * 8),
                key_per_thread,
                width,
            );
            (input, buffer) = (buffer, input);
        }
    };
    input
}

#[cube]
pub trait RadixSort: CubeType + Sized {
    fn one_sweep(
        #[comptime] min_wave_width: u32,
        input: &Array<Self>,
        output: &mut Array<Self>,
        global_histogram: &Array<Atomic<u32>>,
        index: &Atomic<u32>,
        #[comptime] key_per_thread: u32,
        radix_shift: u32,
    );
}

#[cube]
impl RadixSort for u32 {
    fn one_sweep(
        #[comptime] wave_width: u32, // or min wave width
        input: &Array<u32>,
        output: &mut Array<u32>,
        global_histogram: &Array<Atomic<u32>>,
        index: &Atomic<u32>,
        #[comptime] key_per_thread: u32,
        radix_shift: u32,
    ) {
        let log_wave = u32::find_first_set(PLANE_DIM) - 1;

        let radix_mask = comptime! {0xFFu32};
        let mut local_histogram = SharedMemory::<u32>::new(256);
        let wave_histograms = SharedMemory::<Atomic<u32>>::new(
            comptime! {(256/wave_width)<<8},
        );
        // clean
        {
            let mut pos = UNIT_POS;
            for _ in 0..(256 >> log_wave) {
                Atomic::store(&wave_histograms[pos], 0);
                pos += 256;
            }
        }
        if UNIT_POS == 0 {
            local_histogram[0] = Atomic::add(index, 1);
        }
        sync_units();
        let partition_index = local_histogram[0];

        let plane_pos = UNIT_POS >> log_wave;
        let start = plane_pos << 8;

        let wave_hist = wave_histograms.slice(start, start + 256);

        let mut keys = Array::<u32>::new(key_per_thread);
        // load
        let mut idx = UNIT_POS_PLANE
            + plane_pos * (key_per_thread << log_wave)
            + partition_index * key_per_thread * CUBE_DIM;

        #[unroll]
        for i in 0..key_per_thread {
            keys[i] = if idx < input.len() {
                input[idx]
            } else {
                0x_FF_FF_FF_FFu32
            };
            idx += PLANE_DIM;
        }

        let mut offsets = Array::<u32>::new(key_per_thread);
        {
            #[unroll]
            for i in 0..key_per_thread {
                let mut warp_flags =
                    Line::empty(4).fill(0xFF_FF_FF_FFu32);

                #[unroll]
                for k in 0..8 {
                    let cond =
                        (keys[i] >> (k + radix_shift)) & 1 == 1;
                    warp_flags &= if cond {
                        Line::empty(4).fill(0u32)
                    } else {
                        Line::empty(4).fill(0xFF_FF_FF_FFu32)
                    } ^ plane_ballot(cond);
                }
                warp_flags &= count_to_mask(PLANE_DIM);

                let bits = sum_line(Line::count_ones(
                    warp_flags & count_to_mask(UNIT_POS_PLANE),
                ));
                let pre_increment_val = if bits == 0 {
                    Atomic::add(
                        &wave_hist
                            [(keys[i] >> radix_shift) & radix_mask],
                        sum_line(Line::count_ones(warp_flags)),
                    )
                } else {
                    0u32
                };

                offsets[i] = plane_shuffle(
                    pre_increment_val,
                    ffs(warp_flags) - 1,
                ) + bits;
            }
            sync_units();

            {
                let mut reducetion = 0;
                let mut i = UNIT_POS;

                for _ in 0..(CUBE_DIM >> log_wave) {
                    let old = Atomic::load(&wave_histograms[i]);
                    Atomic::store(&wave_histograms[i], reducetion);
                    reducetion += old;
                    i += 256;
                }

                Atomic::add(
                    &global_histogram
                        [UNIT_POS + ((partition_index + 1) << 8)],
                    reducetion | FLAG_REDUCTION,
                );

                local_histogram[UNIT_POS] = ex_block_reduce::<u32>(
                    reducetion, 256u32, wave_width,
                );
            }
            sync_units();
            {
                #[unroll]
                for i in 0..key_per_thread {
                    let t2 = (keys[i] >> radix_shift) & radix_mask;
                    offsets[i] += Atomic::load(&wave_hist[t2])
                        + local_histogram[t2];
                }
            }
            sync_units();
            // maybe locally ordered data is more advantageous for global scatter
            // not generic
            // for i in 0..key_per_thread {
            //     Atomic::store(&wave_histograms[offsets[i]], keys[i])
            // }

            // lookback
            let mut k = partition_index;
            let mut reducetion = 0u32;
            loop {
                let flag_payload = Atomic::load(
                    &global_histogram[(k << 8) + UNIT_POS],
                );
                if (flag_payload & FLAG_MASK) == FLAG_INCLUSIVE {
                    reducetion += flag_payload & VALUE_MASK;
                    Atomic::add(
                        &global_histogram
                            [UNIT_POS + ((partition_index + 1) << 8)],
                        reducetion | FLAG_REDUCTION,
                    );
                    local_histogram[UNIT_POS] =
                        reducetion - local_histogram[UNIT_POS];
                    break;
                }
                if (flag_payload & FLAG_MASK) == FLAG_REDUCTION {
                    reducetion += flag_payload & VALUE_MASK;
                    k -= 1;
                }
            }
            sync_units();

            if partition_index == CUBE_COUNT - 1 {
                #[unroll]
                for i in 0..key_per_thread {
                    let target_idx = local_histogram
                        [keys[i] >> radix_shift & radix_mask]
                        + offsets[i];
                    if target_idx < input.len() {
                        output[target_idx] = keys[i];
                    }
                }
            } else {
                #[unroll]
                for i in 0..key_per_thread {
                    output[local_histogram
                        [keys[i] >> radix_shift & radix_mask]
                        + offsets[i]] = keys[i];
                }
            }
        }
    }
}

#[cube]
fn in_block_reduce<N: Numeric>(
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
fn ex_block_reduce<N: Numeric>(
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

#[cube]
fn sum_line<N: Numeric>(l: Line<N>) -> N {
    let mut s = l[0];
    #[unroll]
    for i in 1..4 {
        s += l[i];
    }
    s
}

#[cube]
fn ffs(l: Line<u32>) -> u32 {
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
fn count_to_mask(count: u32) -> Line<u32> {
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
