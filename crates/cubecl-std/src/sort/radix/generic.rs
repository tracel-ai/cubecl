use std::mem;

use cubecl_core as cubecl;

use super::util::*;
use super::*;
use cubecl::{frontend::Atomic, prelude::*, server::Handle};

const FLAG_MASK: u32 = 0x_C0_00_00_00u32; // 11
const VALUE_MASK: u32 = 0x_3F_FF_FF_FFu32; // 00

const FLAG_REDUCTION: u32 = 0x_40_00_00_00u32; //Flag value indicating reduction of a partition tile is ready
                                               // 10
const FLAG_INCLUSIVE: u32 = 0x_80_00_00_00u32;

pub fn radix_sort<R: Runtime, N: RadixSort + AsRadix>(
    client: &cubecl::prelude::ComputeClient<R::Server, R::Channel>,
    len: usize,
    mut data: Handle,
    buffer: Option<Handle>,
) {
    let mut buffer = buffer.unwrap_or_else(|| client.empty(len));

    let width =
        client.properties().hardware_properties().plane_size_min;
    let offset_align =
        client.properties().memory_properties().alignment as usize;

    let block_size = 256;
    let radix = 256;

    let bytecnt = N::RadixType::BYTECNT as usize;

    let key_per_thread = 16;
    let key_per_block = (block_size * key_per_thread) as usize;

    let block_cnt =
        ((len + key_per_block - 1) / key_per_block) as u32;

    unsafe {
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

        global_histogram::launch::<N, R>(
            client,
            CubeCount::Static(block_cnt, 1, 1),
            CubeDim::new(block_size, 1, 1),
            ArrayArg::from_raw_parts::<N>(&data, len, 1),
            ArrayArg::from_raw_parts::<Atomic<u32>>(
                &global_histogram_handle,
                radix * bytecnt,
                1,
            ),
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

        let index_handle = client.create(u32::as_bytes(&vec![
                0u32;
                bytecnt * (offset_align /size_of::<u32>()).max(1)
            ]));

        for i in 0..bytecnt {
            let offset = (block_cnt + 1) as usize
                * radix
                * i
                * size_of::<u32>();
            let global_histogram_handle = global_histogram_handle
                .clone()
                .offset_start(offset as u64);
            let index_handle = index_handle.clone().offset_start(
                (i * (offset_align / size_of::<u32>()).max(1)
                    * size_of::<u32>()) as u64,
            );

            one_sweep::launch::<N, R>(
                client,
                CubeCount::Static(block_cnt, 1, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<N>(&data, len, 1),
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
                width,
                key_per_thread,
            );
            (data, buffer) = (buffer, data);
        }
    };
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
        comptime! {N::RadixType::BYTECNT<<8},
    );
    // clean mem
    #[unroll]
    for i in 0..N::RadixType::BYTECNT {
        Atomic::store(&shared[UNIT_POS + i * 256], 0);
    }
    sync_units();
    let mut ele_pos = CUBE_POS * CUBE_DIM * key_per_thread + UNIT_POS;
    #[unroll]
    for _ in 0..key_per_thread {
        if ele_pos < len {
            let item = N::as_radix(&input[ele_pos]);
            #[unroll]
            for offset in 0..N::RadixType::BYTECNT {
                Atomic::add(
                    &shared[N::RadixType::shift_mask(
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
    for i in 0..N::RadixType::BYTECNT {
        Atomic::add(
            &global_histogram
                [UNIT_POS + (((block_cnt + 1) * i) << 8)],
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
    input[pos] =
        cube_ex_scan::<u32>(input[pos], cube_size, wave_width)
            | FLAG_INCLUSIVE;
}

#[cube(launch)]
fn one_sweep<N: RadixSort + CubePrimitive>(
    input: &Array<N>,
    output: &mut Array<N>,
    global_histogram: &Array<Atomic<u32>>,
    index: &Array<Atomic<u32>>,
    radix_shift: u32,
    #[comptime] min_wave_width: u32,
    #[comptime] key_per_thread: u32,
) {
    N::one_sweep(
        input,
        output,
        global_histogram,
        &index[0],
        radix_shift,
        min_wave_width,
        key_per_thread,
    );
}

#[cube]
pub trait RadixSort: CubeType + Sized {
    fn one_sweep(
        input: &Array<Self>,
        output: &mut Array<Self>,
        global_histogram: &Array<Atomic<u32>>,
        index: &Atomic<u32>,
        radix_shift: u32,
        #[comptime] min_wave_width: u32,
        #[comptime] key_per_thread: u32,
    );
}

#[cube]
impl<T: AsRadix> RadixSort for T {
    fn one_sweep(
        input: &Array<T>,
        output: &mut Array<T>,
        global_histogram: &Array<Atomic<u32>>,
        index: &Atomic<u32>,
        radix_shift: u32,
        #[comptime] min_wave_width: u32,
        #[comptime] key_per_thread: u32,
    ) {
        let log_wave = u32::find_first_set(PLANE_DIM) - 1;

        let radix_mask = comptime! {0xFFu32};
        let mut local_histogram = SharedMemory::<u32>::new(256);
        let wave_histograms = SharedMemory::<Atomic<u32>>::new(
            comptime! {(256/min_wave_width)<<8},
        );
        // clean
        let mut pos = UNIT_POS;
        for _ in 0..(256 >> log_wave) {
            Atomic::store(&wave_histograms[pos], 0);
            pos += 256;
        }

        if UNIT_POS == 0 {
            local_histogram[0] = Atomic::add(index, 1);
        }
        sync_units();
        let partition_index = local_histogram[0];

        let plane_pos = UNIT_POS >> log_wave;
        let start = plane_pos << 8;

        let wave_hist = wave_histograms.slice(start, start + 256);

        let mut keys = Array::<T>::new(key_per_thread);

        // load
        let mut idx = UNIT_POS_PLANE
            + plane_pos * (key_per_thread << log_wave)
            + partition_index * key_per_thread * CUBE_DIM;

        #[unroll]
        for i in 0..key_per_thread {
            keys[i] = if idx < input.len() {
                input[idx]
            } else {
                T::max_value()
            };
            idx += PLANE_DIM;
        }

        let mut offsets = Array::<u32>::new(key_per_thread);
        #[unroll]
        for i in 0..key_per_thread {
            let mut warp_flags =
                Line::empty(4).fill(0xFF_FF_FF_FFu32);

            #[unroll]
            for k in 0..8 {
                let cond = T::RadixType::shift_mask(
                    &T::as_radix(&keys[i]),
                    k + radix_shift,
                    1,
                ) == 1;
                warp_flags &= if cond {
                    Line::empty(4).fill(0u32)
                } else {
                    Line::empty(4).fill(0xFF_FF_FF_FFu32)
                } ^ plane_ballot(cond);
            }
            warp_flags &= count_to_mask(PLANE_DIM);

            let bits = line_sum(Line::count_ones(
                warp_flags & count_to_mask(UNIT_POS_PLANE),
            ));
            let pre_increment_val = if bits == 0 {
                Atomic::add(
                    &wave_hist[T::RadixType::shift_mask(
                        &T::as_radix(&keys[i]),
                        radix_shift,
                        radix_mask,
                    )],
                    line_sum(Line::count_ones(warp_flags)),
                )
            } else {
                0u32
            };

            offsets[i] = plane_shuffle(
                pre_increment_val,
                line_ffs(warp_flags) - 1,
            ) + bits;
        }
        sync_units();

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

        local_histogram[UNIT_POS] =
            cube_ex_scan::<u32>(reducetion, 256u32, min_wave_width);

        sync_units();
        #[unroll]
        for i in 0..key_per_thread {
            let t2 = T::RadixType::shift_mask(
                &T::as_radix(&keys[i]),
                radix_shift,
                radix_mask,
            );
            offsets[i] +=
                Atomic::load(&wave_hist[t2]) + local_histogram[t2];
        }

        sync_units();

        // lookback
        let mut k = partition_index;
        let mut reducetion = 0u32;
        loop {
            let flag_payload =
                Atomic::load(&global_histogram[(k << 8) + UNIT_POS]);
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
                let target_idx =
                    local_histogram[T::RadixType::shift_mask(
                        &T::as_radix(&keys[i]),
                        radix_shift,
                        radix_mask,
                    )] + offsets[i];
                if target_idx < input.len() {
                    output[target_idx] = keys[i];
                }
            }
        } else {
            #[unroll]
            for i in 0..key_per_thread {
                output[local_histogram[T::RadixType::shift_mask(
                    &T::as_radix(&keys[i]),
                    radix_shift,
                    radix_mask,
                )] + offsets[i]] = keys[i];
            }
        }
    }
}
