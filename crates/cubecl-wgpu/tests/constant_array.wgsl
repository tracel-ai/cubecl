@group(0)
@binding(0)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> info: array<u32>;

const arrays_0: array<f32, 3> = array(f32(3u), f32(5u), f32(1u));

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn main(
) {
    let _2 = arrays_0[2];
    output_0_global[0] = _2;
}