//! Buffers of `#[repr(C)]` structs: kernels index `&[T]` and read or write whole records or
//! single fields.

use alloc::{vec, vec::Vec};
use core::mem::{offset_of, size_of, size_of_val};

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::{ElemType, FloatKind};
use cubecl_runtime::runtime::Runtime;

/// A pixel observation: two coordinates and two indices.
#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Obs {
    pub x: f32,
    pub y: f32,
    pub cam: u32,
    pub point: u32,
}

#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cube]
impl Vec3 {
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn scale(&mut self, k: f32) {
        self.x *= k;
        self.y *= k;
        self.z *= k;
    }
}

/// A record holding a record.
#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Cam {
    pub t: Vec3,
    pub f: f32,
}

/// An f32 then an f64: four bytes of padding before `b`.
#[derive(CubeType, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Padded {
    pub a: f32,
    pub b: f64,
}

#[derive(CubeType, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Tagged<F: Float> {
    pub value: F,
    pub tag: u32,
}

#[derive(CubeType, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ArrayRecord {
    pub values: [u32; 2],
    pub tag: u32,
}

#[cube(launch)]
pub fn record_read(obs: &[Obs], out: &mut [f32]) {
    let i = UNIT_POS as usize;
    let o = obs[i];
    out[i] = o.x + o.y * f32::cast_from(o.cam) + f32::cast_from(obs[i].point) * 100.0;
}

#[cube(launch)]
pub fn record_write_field(obs: &mut [Obs]) {
    let i = UNIT_POS as usize;
    obs[i].y = obs[i].x * 2.0;
    obs[i].point = obs[i].cam + 1;
}

#[cube(launch)]
pub fn record_copy(input: &[Obs], output: &mut [Obs]) {
    let i = UNIT_POS as usize;
    output[i] = input[i];
}

#[cube(launch)]
pub fn record_nested(cams: &[Cam], out: &mut [f32]) {
    let i = UNIT_POS as usize;
    let t = cams[i].t;
    out[i] = (t.x + t.y + t.z) * cams[i].f;
}

#[cube(launch)]
pub fn record_padded(input: &[Padded], out: &mut [f64]) {
    let i = UNIT_POS as usize;
    out[i] = input[i].b + f64::cast_from(input[i].a);
}

#[cube(launch)]
pub fn record_len(obs: &[Obs], out: &mut [u32]) {
    if UNIT_POS == 0 {
        out[0] = obs.len() as u32;
    }
}

#[cube(launch)]
pub fn record_generic<F: Float + HostShareable>(input: &[Tagged<F>], out: &mut [F]) {
    let i = UNIT_POS as usize;
    out[i] = input[i].value * F::cast_from(input[i].tag);
}

#[cube]
fn add_vec3(acc: &mut Vec3, v: &Vec3) {
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
}

#[cube]
fn update_cam(cam: &mut Cam, by: &Cam) {
    cam.f *= by.f;
    cam.t.z *= 2.0;
}

/// References into buffers passed to `#[cube]` functions: to a record field, then to whole
/// records.
#[cube(launch)]
pub fn record_refs(input: &[Cam], output: &mut [Cam]) {
    let i = UNIT_POS as usize;
    add_vec3(&mut output[i].t, &input[i].t);
    update_cam(&mut output[i], &input[i]);
}

#[cube]
fn add3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

/// Records and fields by value, methods, compound assignment, a nested field written, and a
/// local copy written back.
#[cube(launch)]
pub fn record_forms(input: &[Cam], output: &mut [Cam]) {
    let i = UNIT_POS as usize;
    output[i].t = add3(input[i].t, output[i].t);
    output[i].f += input[i].t.dot(&output[i].t);
    let t = &mut output[i].t;
    t.scale(2.0);
    output[i].t.x = 1.0;
    let mut c = output[i];
    c.f *= 0.5;
    output[i] = c;
}

#[allow(unconditional_panic)]
#[cube(launch)]
pub fn record_array_field_index(input: &[ArrayRecord], out: &mut [u32]) {
    let i = UNIT_POS as usize;
    out[2 * i] = input[i].tag;
    out[2 * i + 1] = input[i].values[i];
}

#[allow(unconditional_panic)]
#[cube(launch)]
pub fn record_array_field_assign(records: &mut [ArrayRecord]) {
    let i = UNIT_POS as usize;
    records[i].values[i] = 100 + UNIT_POS;
}

/// The bytes of `records`. `T` must have no padding.
fn bytes_of<T: Copy>(records: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(records.as_ptr().cast(), size_of_val(records)) }
}

fn records_from<T: Copy>(bytes: &[u8]) -> Vec<T> {
    bytes
        .chunks_exact(size_of::<T>())
        .map(|record| unsafe { record.as_ptr().cast::<T>().read_unaligned() })
        .collect()
}

const OBS: [Obs; 2] = [
    Obs {
        x: 1.0,
        y: 2.0,
        cam: 3,
        point: 4,
    },
    Obs {
        x: 5.0,
        y: 6.0,
        cam: 7,
        point: 8,
    },
];

pub fn test_record_read<R: Runtime>(client: Client) {
    let input = client.create_from_slice(bytes_of(&OBS));
    let output = client.empty(2 * size_of::<f32>());

    record_read::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), &[407.0, 847.0]);
}

pub fn test_record_write_field<R: Runtime>(client: Client) {
    let buffer = client.create_from_slice(bytes_of(&OBS));

    record_write_field::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(buffer.clone(), 2) },
    );

    let actual = records_from::<Obs>(&client.read_one_unchecked(buffer));
    let expected = [
        Obs {
            y: 2.0,
            point: 4,
            ..OBS[0]
        },
        Obs {
            y: 10.0,
            point: 8,
            ..OBS[1]
        },
    ];
    assert_eq!(actual, expected);
}

pub fn test_record_copy<R: Runtime>(client: Client) {
    let input = client.create_from_slice(bytes_of(&OBS));
    let output = client.empty(size_of_val(&OBS));

    record_copy::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = records_from::<Obs>(&client.read_one_unchecked(output));
    assert_eq!(actual, OBS);
}

pub fn test_record_nested<R: Runtime>(client: Client) {
    let cams = [
        Cam {
            t: Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            f: 2.0,
        },
        Cam {
            t: Vec3 {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
            f: 0.5,
        },
    ];
    let input = client.create_from_slice(bytes_of(&cams));
    let output = client.empty(2 * size_of::<f32>());

    record_nested::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), &[12.0, 7.5]);
}

pub fn test_record_padded<R: Runtime>(client: Client) {
    if !client
        .properties()
        .supports_type(ElemType::Float(FloatKind::F64))
    {
        return;
    }
    let records = [Padded { a: 1.5, b: 2.25 }, Padded { a: -1.0, b: 1e10 }];
    // Field by field: the padding bytes of `Padded` are not initialized.
    let mut bytes = vec![0u8; size_of_val(&records)];
    for (i, record) in records.iter().enumerate() {
        let base = i * size_of::<Padded>();
        let a = base + offset_of!(Padded, a);
        let b = base + offset_of!(Padded, b);
        bytes[a..a + 4].copy_from_slice(&record.a.to_ne_bytes());
        bytes[b..b + 8].copy_from_slice(&record.b.to_ne_bytes());
    }
    let input = client.create_from_slice(&bytes);
    let output = client.empty(2 * size_of::<f64>());

    record_padded::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f64::from_bytes(&actual), &[3.75, 1e10 - 1.0]);
}

pub fn test_record_len<R: Runtime>(client: Client) {
    let records = [OBS[0], OBS[1], OBS[0]];
    let input = client.create_from_slice(bytes_of(&records));
    let output = client.empty(size_of::<u32>());

    record_len::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(input, 3) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&actual), &[3]);
}

pub fn test_record_generic<R: Runtime>(client: Client) {
    let records = [
        Tagged::<f32> { value: 1.5, tag: 2 },
        Tagged::<f32> {
            value: -0.25,
            tag: 8,
        },
    ];
    let input = client.create_from_slice(bytes_of(&records));
    let output = client.empty(2 * size_of::<f32>());

    record_generic::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), &[3.0, -2.0]);
}

fn cam(t: [f32; 3], f: f32) -> Cam {
    Cam {
        t: Vec3 {
            x: t[0],
            y: t[1],
            z: t[2],
        },
        f,
    }
}

pub fn test_record_refs<R: Runtime>(client: Client) {
    let input = [cam([1.0, 2.0, 3.0], 2.0), cam([4.0, 5.0, 6.0], 0.5)];
    let output = [cam([10.0, 20.0, 30.0], 1.5), cam([0.5, 0.25, 0.125], 4.0)];
    let input = client.create_from_slice(bytes_of(&input));
    let output = client.create_from_slice(bytes_of(&output));

    record_refs::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    let actual = records_from::<Cam>(&client.read_one_unchecked(output));
    let expected = [cam([11.0, 22.0, 66.0], 3.0), cam([4.5, 5.25, 12.25], 2.0)];
    assert_eq!(actual, expected);
}

pub fn test_record_forms<R: Runtime>(client: Client) {
    let input = [cam([1.0, 2.0, 3.0], 2.0), cam([4.0, 5.0, 6.0], 0.5)];
    let start = [cam([10.0, 20.0, 30.0], 1.5), cam([0.5, 0.25, 0.125], 4.0)];
    let input_handle = client.create_from_slice(bytes_of(&input));
    let output = client.create_from_slice(bytes_of(&start));

    record_forms::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input_handle, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 2) },
    );

    // The kernel's steps, on the host.
    let expected: Vec<Cam> = input
        .iter()
        .zip(&start)
        .map(|(a, o)| {
            let t = [a.t.x + o.t.x, a.t.y + o.t.y, a.t.z + o.t.z];
            let f = o.f + (a.t.x * t[0] + a.t.y * t[1] + a.t.z * t[2]);
            cam([1.0, t[1] * 2.0, t[2] * 2.0], f * 0.5)
        })
        .collect();
    let actual = records_from::<Cam>(&client.read_one_unchecked(output));
    assert_eq!(actual, expected);
}

pub fn test_record_array_field_index<R: Runtime>(client: Client) {
    let records = [
        ArrayRecord {
            values: [17, 19],
            tag: 1,
        },
        ArrayRecord {
            values: [23, 29],
            tag: 2,
        },
    ];
    let input = client.create_from_slice(bytes_of(&records));
    let output = client.empty(4 * size_of::<u32>());

    record_array_field_index::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(input, 2) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 4) },
    );

    let actual = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&actual), &[1, 17, 2, 29]);
}

pub fn test_record_array_field_assign<R: Runtime>(client: Client) {
    let records = [
        ArrayRecord {
            values: [17, 19],
            tag: 1,
        },
        ArrayRecord {
            values: [23, 29],
            tag: 2,
        },
    ];
    let buffer = client.create_from_slice(bytes_of(&records));

    record_array_field_assign::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(buffer.clone(), 2) },
    );

    let actual = records_from::<ArrayRecord>(&client.read_one_unchecked(buffer));
    let expected = [
        ArrayRecord {
            values: [100, 19],
            tag: 1,
        },
        ArrayRecord {
            values: [23, 101],
            tag: 2,
        },
    ];
    assert_eq!(actual, expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_record {
    () => {
        mod structs {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_record_read() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_read::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_write_field() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_write_field::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_forms() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_forms::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_refs() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_refs::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_copy() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_copy::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_nested() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_nested::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_padded() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_padded::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_len() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_len::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_generic() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_generic::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_array_field_index() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_array_field_index::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_record_array_field_assign() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::structs::test_record_array_field_assign::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
