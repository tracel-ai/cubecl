@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<vec4<f32>>;

@group(0)
@binding(1)
var<storage, read_write> input_1_global: array<vec4<f32>>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<vec4<f32>>;

@group(0)
@binding(3)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let rank: u32 = info[0];
let _0 = arrayLength(&output_0_global);
let _1 = id < _0;
if _1 {

for (var l_2_0: u32 = 0u; l_2_0 < 256u; l_2_0++) {
let _2 = l_2_0 % 2u;
let _3 = _2 == 0u;
if _3 {
let _4 = input_0_global[id];
let _5 = input_1_global[id];
let _6 = _4 * _5;
let _7 = cos(_6);
let _8 = output_0_global[id];
let _9 = _8 - _7;
output_0_global[id] = _9;
} else {
let _10 = input_0_global[id];
let _11 = input_1_global[id];
let _12 = _10 * _11;
let _13 = cos(_12);
let _14 = output_0_global[id];
let _15 = _14 + _13;
output_0_global[id] = _15;
}
}
}
}