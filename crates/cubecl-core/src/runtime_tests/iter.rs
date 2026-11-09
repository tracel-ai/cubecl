use crate::{self as cubecl};
use alloc::{vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

#[cube(launch)]
fn iter_zip_map_for_each<F: Float>(a: &[F], b: &[F], output: &mut [F]) {
    let scale = F::new(2.0);
    output
        .iter_mut()
        .zip(a.iter().zip(b.iter()).map(|(x, y)| *x + *y * scale))
        .for_each(|(out, value)| *out = value);
}

#[cube(launch)]
fn iter_zip_slice<F: Float>(a: &[F], b: &[F], output: &mut [F]) {
    output[0] = a
        .iter()
        .zip(b)
        .fold(F::new(0.0), |accum, (x, y)| accum + *x * *y);
}

#[cube(launch)]
fn iter_fold(input: &[u32], output: &mut [u32]) {
    output[0] = input.iter().fold(0u32, |accum, x| max(accum, *x));
    let (sum, count) = input
        .iter()
        .fold((0u32, 0u32), |(sum, count), x| (sum + *x, count + 1u32));
    output[1] = sum;
    output[2] = count;
    let mut captured = 0u32;
    input.iter().for_each(|x| captured += *x);
    output[3] = captured;
}

#[cube(launch)]
fn iter_filter(input: &[u32], output: &mut [u32]) {
    let threshold = 2u32;
    output[0] = input.iter().filter(|x| **x % 2u32 == 0u32).count() as u32;
    output[1] = input
        .iter()
        .map(|x| *x * 2u32)
        .filter(|x| *x > threshold)
        .count() as u32;
    output[2] = input
        .iter()
        .copied()
        .filter(|x| *x > threshold)
        .sum::<u32>();
    output[3] = input
        .iter()
        .enumerate()
        .filter(|(i, _)| *i > 0usize)
        .map(|(_, x)| *x)
        .filter(|x| *x < 4u32)
        .sum::<u32>();
    output[4] = input
        .iter()
        .filter(|x| **x > threshold)
        .copied()
        .sum::<u32>();
}

#[cube(launch)]
fn iter_enumerate_rev(input: &[u32], output: &mut [u32]) {
    let len = input.len();
    input
        .iter()
        .rev()
        .enumerate()
        .for_each(|(i, x)| output[i] = *x);
    input
        .iter()
        .enumerate()
        .rev()
        .take(2)
        .for_each(|(i, x)| output[len + i] = *x + 100u32);
}

#[cube(launch)]
fn iter_take_skip(input: &[u32], output: &mut [u32]) {
    let past_end = 100usize;
    let runtime = input[0] as usize;
    output[0] = input.iter().take(2).sum::<u32>();
    output[1] = input.iter().skip(2).sum::<u32>();
    output[2] = input.iter().skip(past_end).count() as u32;
    output[3] = input.iter().take(past_end).count() as u32;
    output[4] = input.iter().skip(1).take(2).product::<u32>();
    output[5] = input.iter().take(runtime).sum::<u32>();
    output[6] = input.iter().skip(1).rev().take(1).sum::<u32>();
    output[7] = input.iter().skip(1).take(2).rev().len() as u32;
}

#[allow(clippy::manual_contains, reason = "Exercise iterator any lowering")]
#[cube(launch)]
fn iter_any_all(input: &[u32], output: &mut [u32]) {
    let present = 3u32;
    let absent = 99u32;
    output[0] = u32::cast_from(input.iter().any(|x| *x == present));
    output[1] = u32::cast_from(input.iter().any(|x| *x == absent));
    output[2] = u32::cast_from(input.iter().all(|x| *x > 0u32));
    output[3] = u32::cast_from(input.iter().all(|x| *x > 1u32));
    output[4] = u32::cast_from(input.iter().filter(|x| **x > 2u32).any(|x| *x == 2u32));
}

/// Counts how many items the predicate sees, to pin that `any` and `all` stop early.
#[cube(launch)]
fn iter_short_circuit(input: &[u32], output: &mut [u32]) {
    let needle = 2u32;
    let mut any_seen = 0u32;
    let found = input.iter().any(|x| {
        any_seen += 1u32;
        *x == needle
    });
    output[0] = any_seen;
    output[1] = u32::cast_from(found);

    let mut all_seen = 0u32;
    let holds = input.iter().all(|x| {
        all_seen += 1u32;
        *x < needle
    });
    output[2] = all_seen;
    output[3] = u32::cast_from(holds);
}

#[cube(launch)]
fn iter_range(input: &[u32], output: &mut [u32]) {
    let len = input.len();
    let low = 2u32;
    let high = 5u32;
    output[0] = (0..len).map(|i| input[i] * 2u32).sum::<u32>();
    output[1] = (high..low).count() as u32;
    output[2] = (low..high).rev().fold(0u32, |accum, x| accum * 10u32 + x);
    output[3] = (0..len)
        .zip(input.iter())
        .map(|(i, x)| i as u32 * *x)
        .sum::<u32>();
    output[4] = (low..=high).rev().fold(0u32, |accum, x| accum * 10u32 + x);
    output[5] = (low..high).skip(1).len() as u32;
    output[6] = range_stepped(0u32, 7u32, 3u32)
        .map(|x| x + 1u32)
        .sum::<u32>();
    // A constant length of one must still be a loop that `any` can break out of.
    output[7] = u32::cast_from((0u32..1u32).any(|x| x == 0u32));
    // An inclusive range can end at the largest value.
    let max = 4294967295u32;
    output[8] = ((max - 1u32)..=max).rev().count() as u32;
    output[9] = ((max - 1u32)..=max).rev().map(|x| max - x).sum::<u32>();
    output[10] = (high..=low).rev().count() as u32;
}

#[cube(launch)]
fn iter_signed_range(output: &mut [i32]) {
    let low = -2i32;
    let high = 2i32;
    output[0] = (low..high).map(|x| x * 3i32).sum::<i32>();
    output[1] = (high..low).count() as i32;
    output[2] = (low..high).rev().fold(0i32, |accum, x| accum * 10i32 + x);
}

#[allow(
    clippy::iter_count,
    clippy::manual_contains,
    reason = "Exercise iterator count and any on empty subslices"
)]
#[cube(launch)]
fn iter_subslice(input: &[u32], output: &mut [u32]) {
    let len = input.len();
    output[0] = input[1..3].iter().sum::<u32>();
    output[1] = input[len..len].iter().count() as u32;
    output[2] = input[len..len].iter().sum::<u32>();
    output[3] = input[len..len].iter().rev().sum::<u32>();
    output[4] = input[len..len].iter().product::<u32>();
    output[5] = u32::cast_from(input[len..len].iter().all(|x| *x == 77u32));
    output[6] = u32::cast_from(input[len..len].iter().any(|x| *x == 77u32));
}

#[cube(launch)]
fn iter_composition(input: &[u32], output: &mut [u32]) {
    // A chain in a closure of another chain is a nested loop.
    output[0] = input
        .iter()
        .map(|x| input.iter().map(|y| *x * *y).sum::<u32>())
        .sum::<u32>();
    // A chain can be stored before it's consumed.
    let doubled = input.iter().map(|x| *x * 2u32);
    output[1] = doubled.sum::<u32>();
    // A chain is an `Iterable`, so a `for` loop can consume it.
    let mut total = 0u32;
    for value in input.iter().rev().map(|x| *x + 1u32) {
        total = total * 10u32 + value;
    }
    output[2] = total;
    // An array iterates through its slice.
    let mut array = Array::<u32>::new(3usize);
    array
        .iter_mut()
        .enumerate()
        .for_each(|(i, value)| *value = i as u32 + 1u32);
    output[3] = array.iter().sum::<u32>();
}

#[cube(launch)]
fn iter_mut_subslice(output: &mut [u32]) {
    output[2..4].iter_mut().for_each(|x| *x += 10u32);
    output
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| *i < 1usize)
        .for_each(|(_, x)| *x = 7u32);
}

fn run_u32(
    client: &Client,
    input: &[u32],
    outputs: usize,
    launch: impl FnOnce(BufferArg, BufferArg),
) -> Vec<u32> {
    let input_handle = client.create_from_slice(u32::as_bytes(input));
    let output = zeroed::<u32>(client, outputs);
    launch(
        unsafe { BufferArg::from_raw_parts(input_handle, input.len()) },
        unsafe { BufferArg::from_raw_parts(output.clone(), outputs) },
    );
    u32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

fn zeroed<T: CubeElement + Default + Clone>(client: &Client, len: usize) -> Handle {
    client.create_from_slice(T::as_bytes(&vec![T::default(); len]))
}

macro_rules! launch_u32 {
    ($client:expr, $kernel:ident, $input:expr, $outputs:expr) => {
        run_u32(&$client, $input, $outputs, |input, output| {
            $kernel::launch(
                &$client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                input,
                output,
            )
        })
    };
}

const INPUT: [u32; 4] = [1, 2, 3, 4];

pub fn test_iter_zip_map_for_each(client: Client) {
    let a = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0]));
    let b = client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let output = zeroed::<f32>(&client, 5);

    iter_zip_map_for_each::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(a, 3) },
        unsafe { BufferArg::from_raw_parts(b, 4) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 5) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), &[21.0, 42.0, 63.0, 0.0, 0.0]);
}

pub fn test_iter_zip_slice(client: Client) {
    let a = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0]));
    let b = client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let output = zeroed::<f32>(&client, 1);

    iter_zip_slice::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(a, 3) },
        unsafe { BufferArg::from_raw_parts(b, 4) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), &[140.0]);
}

pub fn test_iter_fold(client: Client) {
    let actual = launch_u32!(client, iter_fold, &INPUT, 4);
    assert_eq!(actual, [4, 10, 4, 10]);
}

pub fn test_iter_filter(client: Client) {
    let actual = launch_u32!(client, iter_filter, &INPUT, 5);
    assert_eq!(actual, [2, 3, 7, 5, 7]);
}

pub fn test_iter_enumerate_rev(client: Client) {
    let actual = launch_u32!(client, iter_enumerate_rev, &INPUT, 8);
    assert_eq!(actual, [4, 3, 2, 1, 0, 0, 103, 104]);
}

pub fn test_iter_take_skip(client: Client) {
    let actual = launch_u32!(client, iter_take_skip, &[3, 2, 3, 4], 8);
    assert_eq!(actual, [5, 7, 0, 4, 6, 8, 4, 2]);
}

pub fn test_iter_any_all(client: Client) {
    let actual = launch_u32!(client, iter_any_all, &INPUT, 5);
    assert_eq!(actual, [1, 0, 1, 0, 0]);
}

pub fn test_iter_short_circuit(client: Client) {
    let actual = launch_u32!(client, iter_short_circuit, &INPUT, 4);
    assert_eq!(actual, [2, 1, 2, 0]);
}

pub fn test_iter_range(client: Client) {
    let actual = launch_u32!(client, iter_range, &INPUT, 11);
    // Item 3 is 0*1 + 1*2 + 2*3 + 3*4, and item 6 is 1 + 4 + 7.
    assert_eq!(actual, [20, 0, 432, 20, 5432, 2, 12, 1, 2, 1, 0]);
}

pub fn test_iter_signed_range(client: Client) {
    let output = zeroed::<i32>(&client, 3);

    iter_signed_range::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(output.clone(), 3) },
    );

    let actual = client.read_one_unchecked(output);
    // Item 2 folds 1, 0, -1, -2 into 1, 10, 99, 988.
    assert_eq!(i32::from_bytes(&actual), &[-6, 0, 988]);
}

pub fn test_iter_subslice(client: Client) {
    let actual = launch_u32!(client, iter_subslice, &INPUT, 7);
    assert_eq!(actual, [5, 0, 0, 0, 1, 1, 0]);
}

pub fn test_iter_composition(client: Client) {
    let actual = launch_u32!(client, iter_composition, &INPUT, 4);
    assert_eq!(actual, [100, 20, 5432, 6]);
}

pub fn test_iter_mut_subslice(client: Client) {
    let output = client.create_from_slice(u32::as_bytes(&[1, 2, 3, 4, 5]));

    iter_mut_subslice::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(output.clone(), 5) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&actual), &[7, 2, 13, 14, 5]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_iter {
    () => {
        use super::*;

        $crate::testgen_iter!(@tests
            test_iter_zip_map_for_each,
            test_iter_zip_slice,
            test_iter_fold,
            test_iter_filter,
            test_iter_enumerate_rev,
            test_iter_take_skip,
            test_iter_any_all,
            test_iter_short_circuit,
            test_iter_range,
            test_iter_signed_range,
            test_iter_subslice,
            test_iter_composition,
            test_iter_mut_subslice,
        );
    };
    (@tests $($name:ident,)*) => {
        $(
            #[$crate::runtime_tests::test_log::test]
            fn $name() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::iter::$name(client);
            }
        )*
    };
}
