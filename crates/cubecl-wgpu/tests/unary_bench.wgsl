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
fn execute_unary_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let _0 = info[5u];
let _1 = id < _0;
if _1 {

for (var l_2_2: u32 = 0u; l_2_2 < 256u; l_2_2++) {
let _3 = l_2_2 % 2u;
let _4 = _3 == 0u;
if _4 {
var l_3_0: u32;
l_3_0 = info[0u];
let _5 = select(vec4<f32>(0), input_0_global[id], id < l_3_0);
var l_3_1: u32;
l_3_1 = info[1u];
let _6 = select(vec4<f32>(0), input_1_global[id], id < l_3_1);
let _7 = _5 * _6;
let _8 = cos(_7);
var l_3_2: u32;
l_3_2 = info[2u];
let _9 = select(vec4<f32>(0), output_0_global[id], id < l_3_2);
let _10 = _9 - _8;
var l_3_3: u32;
var l_3_4: bool;
l_3_3 = info[2u];
l_3_4 = id < l_3_3;
if l_3_4 {
output_0_global[id] = _10;
}
} else {
var l_3_0: u32;
l_3_0 = info[0u];
let _11 = select(vec4<f32>(0), input_0_global[id], id < l_3_0);
var l_3_1: u32;
l_3_1 = info[1u];
let _12 = select(vec4<f32>(0), input_1_global[id], id < l_3_1);
let _13 = _11 * _12;
let _14 = cos(_13);
var l_3_2: u32;
l_3_2 = info[2u];
let _15 = select(vec4<f32>(0), output_0_global[id], id < l_3_2);
let _16 = _15 + _14;
var l_3_3: u32;
var l_3_4: bool;
l_3_3 = info[2u];
l_3_4 = id < l_3_3;
if l_3_4 {
output_0_global[id] = _16;
}
}
}
}
}
