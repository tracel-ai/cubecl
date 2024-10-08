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

for (var l_2_2: u32 = 0u; l_2_2 < 256u; l_2_2++) {
let _3 = l_2_2 % 2u;
let _4 = _3 == 0u;
if _4 {
let _5 = input_0_global[id];
let _6 = input_1_global[id];
let _7 = _5 * _6;
let _8 = cos(_7);
let _9 = output_0_global[id];
let _10 = _9 - _8;
output_0_global[id] = _10;
} else {
let _11 = input_0_global[id];
let _12 = input_1_global[id];
let _13 = _11 * _12;
let _14 = cos(_13);
let _15 = output_0_global[id];
let _16 = _15 + _14;
output_0_global[id] = _16;
}
}
}
}